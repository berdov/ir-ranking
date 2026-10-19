#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Boolean retrieval on a disk-backed, delta-compressed inverted index."""

import argparse
from array import array
import csv
from functools import lru_cache
import os
from pathlib import Path
import sqlite3
import tempfile
from timeit import default_timer as timer

from nltk import tokenize


def preprocess(text):
    """Токенизация и нормализация. НЕ МЕНЯЙТЕ эту функцию:
    ей же размечен эталон на Kaggle."""
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]


INDEX_FILE = "index.sqlite3"
FORMAT_VERSION = "1"
DOCUMENTS_FILE = "vkmarco-docs.tsv"
QUERIES_FILE = "vkmarco-doceval-queries.tsv"


def read_tsv(path, fields):
    """Read raw TSV, preserving quotes, empty fields and tabs in the last field."""
    with Path(path).open(encoding="utf-8", newline="") as source:
        for number, line in enumerate(source, 1):
            row = line.rstrip("\r\n").split("\t", fields - 1)
            if len(row) != fields or not row[0]:
                raise ValueError(f"{path}:{number}: expected {fields} fields and a nonempty ID")
            yield row


def read_csv(path, header):
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        if next(reader, None) != header:
            raise ValueError(f"{path}: expected CSV header {','.join(header)}")
        for row in reader:
            if len(row) != len(header) or not row[0]:
                raise ValueError(f"{path}:{reader.line_num}: malformed CSV record")
            yield row


def encode_postings(postings):
    """Unsigned base-128 varints of gaps between increasing internal document IDs."""
    encoded = bytearray()
    previous = 0
    for doc_id in postings:
        gap = doc_id - previous
        if gap < 0:
            raise ValueError("Posting lists must be sorted")
        while gap >= 128:
            encoded.append((gap & 127) | 128)
            gap >>= 7
        encoded.append(gap)
        previous = doc_id
    return bytes(encoded)


def decode_postings(encoded):
    previous = gap = shift = 0
    for value in encoded:
        gap |= (value & 127) << shift
        if value & 128:
            shift += 7
            if shift > 28:
                raise ValueError("Corrupt posting list: document ID exceeds uint32")
        else:
            previous += gap
            yield previous
            gap = shift = 0
    if shift:
        raise ValueError("Corrupt posting list: truncated varint")


def configure_database(connection, cache_kib=2048):
    # Limit SQLite's page cache and avoid mapping the whole index into RAM.
    connection.execute(f"PRAGMA cache_size = -{cache_kib}")
    connection.execute("PRAGMA mmap_size = 0")
    connection.execute("PRAGMA temp_store = FILE")


def build_index(data_dir, index_dir):
    """The 100 MB limit applies to submission; building keeps compact uint32 arrays."""
    data_dir, index_dir = Path(data_dir), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    # Build privately, then replace the completed index. A failed rebuild keeps
    # the previous usable index intact.
    with tempfile.TemporaryDirectory(prefix=".building-", dir=index_dir) as work:
        database = Path(work) / INDEX_FILE
        connection = sqlite3.connect(database)
        try:
            configure_database(connection, cache_kib=4096)
            connection.executescript("""
                CREATE TABLE documents (
                    external_id TEXT PRIMARY KEY,
                    internal_id INTEGER NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE terms (
                    term TEXT PRIMARY KEY,
                    document_frequency INTEGER NOT NULL,
                    postings BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
            """)
            inverted = {}
            document_count = posting_count = 0
            with connection:
                for external_id, _url, title, body in read_tsv(data_dir / DOCUMENTS_FILE, 4):
                    internal_id = document_count
                    if internal_id > 0xFFFFFFFF:
                        raise ValueError("Too many documents for uint32 posting lists")
                    try:
                        connection.execute("INSERT INTO documents VALUES (?, ?)",
                                           (external_id, internal_id))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Duplicate DocumentId: {external_id}") from error
                    # URL is deliberately excluded. One posting per term per document.
                    for term in set(preprocess(title + " " + body)):
                        if term not in inverted:
                            inverted[term] = array("I")
                        inverted[term].append(internal_id)
                        posting_count += 1
                    document_count += 1
                term_count = len(inverted)
                for term in sorted(inverted):
                    postings = inverted.pop(term)
                    connection.execute("INSERT INTO terms VALUES (?, ?, ?)",
                                       (term, len(postings), encode_postings(postings)))
                connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
                    ("format_version", FORMAT_VERSION),
                    ("document_count", str(document_count)),
                    ("term_count", str(term_count)),
                    ("posting_count", str(posting_count)),
                ])
        finally:
            connection.close()
        os.replace(database, index_dir / INDEX_FILE)
    print(f"indexed documents={document_count}, terms={term_count}, postings={posting_count}")


class BooleanIndex:
    """SQLite stores the vocabulary and IDs; only the current postings are read."""

    def __init__(self, index_dir):
        database = (Path(index_dir) / INDEX_FILE).resolve()
        if not database.is_file():
            raise ValueError(f"Index not found: {database}; run --build_index first")
        self.connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        try:
            configure_database(self.connection)
            self.connection.execute("PRAGMA query_only = ON")
            metadata = dict(self.connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported index format; rebuild the index")
            self.document_count = int(metadata["document_count"])
            self.bitmap_size = (self.document_count + 7) // 8
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def document_id(self, external_id):
        row = self.connection.execute(
            "SELECT internal_id FROM documents WHERE external_id = ?", (external_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown DocumentId in objects.csv: {external_id}")
        return row[0]

    def search(self, tokens):
        """AND query as a compact bitmap; an empty AND matches every document."""
        terms = []
        for term in set(tokens):
            row = self.connection.execute(
                "SELECT document_frequency FROM terms WHERE term = ?", (term,)
            ).fetchone()
            if row is None:
                return bytes(self.bitmap_size)
            terms.append((row[0], term))

        if not terms:
            return ((1 << self.document_count) - 1).to_bytes(self.bitmap_size, "little")

        matches = None
        for _frequency, term in sorted(terms):
            encoded = self.connection.execute(
                "SELECT postings FROM terms WHERE term = ?", (term,)
            ).fetchone()[0]
            bitmap = bytearray(self.bitmap_size)
            for doc_id in decode_postings(encoded):
                if doc_id >= self.document_count:
                    raise ValueError("Corrupt posting list: unknown internal document ID")
                bitmap[doc_id // 8] |= 1 << (doc_id % 8)
            current = int.from_bytes(bitmap, "little")
            matches = current if matches is None else matches & current
            if not matches:
                break
        return matches.to_bytes(self.bitmap_size, "little")


def generate_submission(data_dir, index_dir, submission_file):
    data_dir, submission_file = Path(data_dir), Path(submission_file)
    inputs = [data_dir / name for name in
              (DOCUMENTS_FILE, QUERIES_FILE, "objects.csv", "sample_submission.csv")]
    inputs.append(Path(index_dir) / INDEX_FILE)
    if submission_file.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Submission must not overwrite an input file or the index")
    submission_file.parent.mkdir(parents=True, exist_ok=True)
    index = BooleanIndex(index_dir)
    output_path = None
    try:
        # Intermediate query bitmaps and sample ObjectIds stay on disk, too.
        # Their total size therefore does not determine process memory usage.
        with tempfile.TemporaryDirectory(prefix="boolean-submission-") as work:
            scratch = sqlite3.connect(Path(work) / "queries.sqlite3")
            try:
                configure_database(scratch)
                # These tables are disposable, never the persistent index.
                scratch.execute("PRAGMA journal_mode = OFF")
                scratch.execute("PRAGMA synchronous = OFF")
                scratch.executescript("""
                    CREATE TABLE queries (id TEXT PRIMARY KEY, hits BLOB NOT NULL) WITHOUT ROWID;
                    CREATE TABLE sample_ids (id TEXT PRIMARY KEY) WITHOUT ROWID;
                """)
                with scratch:
                    for query_id, text in read_tsv(data_dir / QUERIES_FILE, 2):
                        try:
                            scratch.execute("INSERT INTO queries VALUES (?, ?)",
                                            (query_id, index.search(preprocess(text))))
                        except sqlite3.IntegrityError as error:
                            raise ValueError(f"Duplicate QueryId: {query_id}") from error
                    for object_id, _random_label in read_csv(
                            data_dir / "sample_submission.csv", ["ObjectId", "Label"]):
                        try:
                            scratch.execute("INSERT INTO sample_ids VALUES (?)", (object_id,))
                        except sqlite3.IntegrityError as error:
                            raise ValueError(f"Duplicate ObjectId in sample: {object_id}") from error

                # At most about 1 MiB of bitmaps plus small key/cache overhead.
                capacity = max(1, min(128, (1024 * 1024) // max(1, index.bitmap_size)))

                @lru_cache(maxsize=capacity)
                def query_hits(query_id):
                    row = scratch.execute("SELECT hits FROM queries WHERE id = ?",
                                          (query_id,)).fetchone()
                    if row is None:
                        raise ValueError(f"Unknown QueryId in objects.csv: {query_id}")
                    return row[0]

                lookup_document = lru_cache(maxsize=4096)(index.document_id)
                count = positives = 0
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="", prefix=".submission-",
                    suffix=".csv", dir=submission_file.parent, delete=False,
                ) as output:
                    output_path = Path(output.name)
                    writer = csv.writer(output, lineterminator="\n")
                    writer.writerow(["ObjectId", "Label"])
                    with scratch:
                        for object_id, query_id, external_id in read_csv(
                            data_dir / "objects.csv", ["ObjectId", "QueryId", "DocumentId"]
                        ):
                            removed = scratch.execute("DELETE FROM sample_ids WHERE id = ?",
                                                      (object_id,)).rowcount
                            if removed != 1:
                                raise ValueError(f"ObjectId {object_id} is duplicated in objects.csv "
                                                 "or absent from sample_submission.csv")
                            internal_id = lookup_document(external_id)
                            hits = query_hits(query_id)
                            label = (hits[internal_id // 8] >> (internal_id % 8)) & 1
                            writer.writerow([object_id, label])
                            count += 1
                            positives += label
                        missing = scratch.execute("SELECT id FROM sample_ids LIMIT 1").fetchone()
                        if missing is not None:
                            raise ValueError(f"Sample ObjectId {missing[0]} is absent from objects.csv")
                query_hits.cache_clear()
                lookup_document.cache_clear()
            finally:
                scratch.close()
        os.replace(output_path, submission_file)
        output_path = None
    finally:
        index.close()
        if output_path is not None:
            output_path.unlink(missing_ok=True)
    print(f"submission objects={count}, positives={positives}, file={submission_file}")


def main():
    parser = argparse.ArgumentParser(description="Indexing homework solution")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build_index", action="store_true", help="force reindexing")
    mode.add_argument("--submission_file", help="output Kaggle submission file")
    parser.add_argument("--index_dir", required=True, help="index directory")
    parser.add_argument("data_dir", help="input data directory")
    args = parser.parse_args()
    start = timer()
    try:
        if args.build_index:
            build_index(args.data_dir, args.index_dir)
        else:
            generate_submission(args.data_dir, args.index_dir, args.submission_file)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"finished, elapsed = {timer() - start:.3f}")


if __name__ == "__main__":
    main()
