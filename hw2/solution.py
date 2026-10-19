#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Classical text reranking: streaming features and train-only parameter search.

    ./solution.py --submission_file=submission.csv PATH-TO-DATA

Ranking formulas are implemented in ranking.py, shipped in the same repository.
No trained rankers, neural networks, external corpora or ready-made BM25 scorers.
"""

import argparse
from array import array
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from time import perf_counter
import unicodedata
from urllib.parse import unquote, urlsplit

import numpy as np

from ranking import RankingData, score, select_parameters


RAW_WORDS = re.compile(r"(?u)\b\w\w+\b")
CYRILLIC = re.compile(r"[а-яё]")
FILE_NAMES = (
    "vkmarco-docs.tsv", "vkmarco-doctrain-queries.tsv", "vkmarco-doctrain-qrels.tsv",
    "vkmarco-doceval-queries.tsv", "objects.csv", "sample_submission.csv",
)


def _without_stress(text):
    # Normalize canonical variants such as U+0341 before removing accents.
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", decomposed.replace("\u0301", "").replace("\u0300", ""))


class Normalizer:
    """Cached Russian lemmatization; no downloads or external corpus at runtime."""

    def __init__(self, mode="lemma"):
        if mode not in {"none", "lemma"}:
            raise ValueError("normalization must be 'none' or 'lemma'")
        self.mode = mode
        self.morph = None
        if mode == "lemma":
            import pymorphy3
            self.morph = pymorphy3.MorphAnalyzer()
        self.word = lru_cache(maxsize=400_000)(self._word)

    def _word(self, token):
        # Preserve й: remove only stress marks, not all combining characters.
        token = _without_stress(token)
        token = token.lower().replace("ё", "е")
        if self.morph is not None and len(token) <= 40 and token.isalpha() and CYRILLIC.search(token):
            return self.morph.parse(token)[0].normal_form.replace("ё", "е")
        return token

    def tokens(self, text):
        text = _without_stress(text)
        return [self.word(token) for token in RAW_WORDS.findall(text.lower())]


def read_queries(path):
    queries = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f"{path}:{line_number}: expected QueryId and QueryText")
            query_id, text = fields
            if query_id in queries:
                raise ValueError(f"Duplicate QueryId in {path}: {query_id}")
            queries[query_id] = text
    return queries


def read_qrels(path):
    qrels = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected QueryId 0 DocumentId Label")
            query_id, _, doc_id, raw_label = fields
            label = int(raw_label)
            if not 0 <= label <= 3:
                raise ValueError(f"{path}:{line_number}: relevance must be between 0 and 3")
            group = qrels.setdefault(query_id, {})
            if doc_id in group:
                raise ValueError(f"Duplicate qrel pair: {query_id}, {doc_id}")
            group[doc_id] = label
    return qrels


def read_csv(path, header):
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        if next(reader, None) != header:
            raise ValueError(f"{path}: expected header {','.join(header)}")
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header) or any(not value for value in row):
                raise ValueError(f"{path}:{line_number}: malformed CSV record")
            yield tuple(row)


def read_objects(path):
    objects = list(read_csv(path, ["ObjectId", "QueryId", "DocumentId"]))
    if len({row[0] for row in objects}) != len(objects):
        raise ValueError("Duplicate ObjectId in objects.csv")
    return objects


def _url_text(url):
    try:
        parts = urlsplit(url)
        return unquote((parts.hostname or "") + " " + parts.path + " " + parts.query)
    except ValueError:
        return unquote(url)


def _phrase_present(sequence, phrase):
    if len(phrase) < 2 or len(sequence) < len(phrase):
        return 0.0
    first = phrase[0]
    width = len(phrase)
    return float(any(token == first and sequence[i:i + width] == phrase
                     for i, token in enumerate(sequence[:len(sequence) - width + 1])))


def _proximity(sequence, wanted):
    """Shortest covering window density; no cross-field windows, singletons 0."""
    if len(wanted) < 2:
        return 0.0
    positions = [(i, token) for i, token in enumerate(sequence) if token in wanted]
    counts = Counter()
    left = covered = 0
    best = math.inf
    for right_position, token in positions:
        if counts[token] == 0:
            covered += 1
        counts[token] += 1
        while covered == len(wanted):
            left_position, left_token = positions[left]
            best = min(best, right_position - left_position + 1)
            counts[left_token] -= 1
            if counts[left_token] == 0:
                covered -= 1
            left += 1
    return len(wanted) / best if math.isfinite(best) else 0.0


def _raw_tokens(text):
    # Equivalent token pattern/lowercasing to default TfidfVectorizer.
    return RAW_WORDS.findall(text.lower())


def build_features(data_dir, normalization="lemma"):
    """Read the full corpus once, keeping sparse raw counts and candidate features.

    Corpus statistics are unsupervised and use every supplied document. All
    label-based choices are deferred to select_parameters and training queries.
    """
    started = perf_counter()
    data_dir = Path(data_dir)
    for filename in FILE_NAMES:
        if not (data_dir / filename).is_file():
            raise ValueError(f"Missing HW2 file: {data_dir / filename}; HW1 data cannot be reused")
    train_queries = read_queries(data_dir / FILE_NAMES[1])
    qrels = read_qrels(data_dir / FILE_NAMES[2])
    test_queries = read_queries(data_dir / FILE_NAMES[3])
    objects = read_objects(data_dir / "objects.csv")
    sample_ids = [row[0] for row in read_csv(data_dir / "sample_submission.csv", ["ObjectId", "Score"])]
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != {row[0] for row in objects}:
        raise ValueError("sample_submission.csv ObjectIds differ from objects.csv or contain duplicates")
    if not qrels or len(qrels) < 2:
        raise ValueError("At least two labelled training queries are required for a query-level holdout")
    if not set(qrels) <= set(train_queries):
        raise ValueError("qrels contain an unknown training QueryId")
    if any(qid not in test_queries for _, qid, _ in objects):
        raise ValueError("objects.csv contains an unknown test QueryId")

    query_text = {"train:" + key: text for key, text in train_queries.items()}
    query_text.update({"test:" + key: text for key, text in test_queries.items()})
    normalizer = Normalizer(normalization)
    query_tokens = {key: normalizer.tokens(text) for key, text in query_text.items()}
    query_counts = {key: Counter(tokens) for key, tokens in query_tokens.items()}
    query_raw = {key: Counter(_raw_tokens(text)) for key, text in query_text.items()}
    vocabulary = sorted({term for tokens in query_tokens.values() for term in tokens})
    term_ids = {term: i for i, term in enumerate(vocabulary)}
    df = np.zeros(len(vocabulary), dtype=np.int64)
    df_url = np.zeros(len(vocabulary), dtype=np.int64)
    cf = np.zeros(len(vocabulary), dtype=np.int64)

    query_ids, doc_ids, labels = [], [], []
    for query_id in sorted(qrels):
        for doc_id in sorted(qrels[query_id]):
            query_ids.append("train:" + query_id)
            doc_ids.append(doc_id)
            labels.append(qrels[query_id][doc_id])
    training_pairs = len(labels)
    for _object_id, query_id, doc_id in objects:
        query_ids.append("test:" + query_id)
        doc_ids.append(doc_id)
        labels.append(-1)
    pairs = len(labels)
    by_document = defaultdict(list)
    for pair, doc_id in enumerate(doc_ids):
        by_document[doc_id].append(pair)
    offsets = np.zeros(pairs + 1, dtype=np.int64)
    for pair, query_id in enumerate(query_ids):
        offsets[pair + 1] = offsets[pair] + len(query_counts[query_id])
    entries = int(offsets[-1])
    term_pair = np.repeat(np.arange(pairs, dtype=np.int32), np.diff(offsets))
    term_index = np.empty(entries, dtype=np.int32)
    qtf = np.empty(entries, dtype=np.float64)
    query_terms = {key: list(counts) for key, counts in query_counts.items()}
    for pair, query_id in enumerate(query_ids):
        start, end = offsets[pair:pair + 2]
        terms = query_terms[query_id]
        term_index[start:end] = [term_ids[term] for term in terms]
        qtf[start:end] = [query_counts[query_id][term] for term in terms]
    tf_title, tf_body, tf_url = (np.zeros(entries, dtype=np.float64) for _ in range(3))
    length_title, length_body, length_url = (np.zeros(pairs, dtype=np.float64) for _ in range(3))
    title_coverage, body_coverage, phrase_title, phrase_body, proximity = (
        np.zeros(pairs, dtype=np.float64) for _ in range(5))
    pair_rows = np.empty(pairs, dtype=np.int32)

    # Raw TF-IDF needs document norms over ALL terms, not only query terms.
    # Compact uint32 arrays avoid millions of Python integer/list objects.
    raw_vocabulary = {}
    raw_df = array("I")
    raw_columns, raw_values = array("I"), array("I")
    raw_offsets = array("Q", [0])
    raw_pair_tf = []
    raw_pair_terms = []
    raw_pair_qtf = []
    raw_pair_index = []
    seen = set()
    total_title = total_body = total_url = 0
    with (data_dir / "vkmarco-docs.tsv").open(encoding="utf-8") as source:
        for row, line in enumerate(source):
            fields = line.rstrip("\r\n").split("\t", 3)
            fields += [""] * (4 - len(fields))
            doc_id, url, title, body = fields
            if not doc_id or doc_id in seen:
                raise ValueError(f"Empty or duplicate DocumentId at corpus line {row + 1}: {doc_id}")
            seen.add(doc_id)
            raw_title, raw_body = _raw_tokens(title), _raw_tokens(body)
            raw_counts = Counter(raw_title)
            raw_counts.update(raw_body)
            for word, count in raw_counts.items():
                word_id = raw_vocabulary.get(word)
                if word_id is None:
                    word_id = len(raw_vocabulary)
                    raw_vocabulary[word] = word_id
                    raw_df.append(0)
                raw_df[word_id] += 1
                raw_columns.append(word_id)
                raw_values.append(count)
            raw_offsets.append(len(raw_columns))

            title_tokens = normalizer.tokens(title)
            body_tokens = normalizer.tokens(body)
            url_tokens = normalizer.tokens(_url_text(url))
            title_counts, body_counts, url_counts = map(Counter, (title_tokens, body_tokens, url_tokens))
            title_len, body_len, url_len = len(title_tokens), len(body_tokens), len(url_tokens)
            total_title += title_len
            total_body += body_len
            total_url += url_len
            relevant = (title_counts.keys() | body_counts.keys()) & term_ids.keys()
            for term in relevant:
                term_id = term_ids[term]
                df[term_id] += 1
                cf[term_id] += title_counts[term] + body_counts[term]
            for term in (relevant | (url_counts.keys() & term_ids.keys())):
                df_url[term_ids[term]] += 1

            for pair in by_document.get(doc_id, ()):
                query_id = query_ids[pair]
                terms = query_terms[query_id]
                start, end = offsets[pair:pair + 2]
                tf_title[start:end] = [title_counts[term] for term in terms]
                tf_body[start:end] = [body_counts[term] for term in terms]
                tf_url[start:end] = [url_counts[term] for term in terms]
                length_title[pair], length_body[pair], length_url[pair] = title_len, body_len, url_len
                pair_rows[pair] = row
                title_coverage[pair] = sum(term in title_counts for term in terms) / max(1, len(terms))
                body_coverage[pair] = sum(term in body_counts for term in terms) / max(1, len(terms))
                phrase_title[pair] = _phrase_present(title_tokens, query_tokens[query_id])
                phrase_body[pair] = _phrase_present(body_tokens, query_tokens[query_id])
                if all(term in body_counts for term in terms):
                    proximity[pair] = _proximity(body_tokens, set(terms))
                for word, frequency in query_raw[query_id].items():
                    if word in raw_counts:
                        raw_pair_index.append(pair)
                        raw_pair_terms.append(raw_vocabulary[word])
                        raw_pair_tf.append(raw_counts[word])
                        raw_pair_qtf.append(frequency)
            if (row + 1) % 10_000 == 0:
                print(f"read documents={row + 1}, elapsed={perf_counter() - started:.1f}s", flush=True)

    missing = by_document.keys() - seen
    if missing:
        raise ValueError(f"Unknown DocumentId referenced by candidates: {sorted(missing)[:3]}")
    documents = len(seen)
    if not documents:
        raise ValueError("Document corpus is empty")
    idf = np.log1p((documents - df + 0.5) / (df + 0.5))
    idf_url = np.log1p((documents - df_url + 0.5) / (df_url + 0.5))
    collection_prob = cf / max(1, total_title + total_body)

    # Exact smoothed TF-IDF: idf=log((1+N)/(1+df))+1, raw TF, L2 norms.
    raw_idf = np.log((1.0 + documents) / (1.0 + np.asarray(raw_df))) + 1.0
    columns = np.frombuffer(raw_columns, dtype=np.uint32)
    values = np.frombuffer(raw_values, dtype=np.uint32)
    norms = np.zeros(documents, dtype=np.float64)
    for row in range(documents):
        start, end = raw_offsets[row:row + 2]
        weighted = values[start:end] * raw_idf[columns[start:end]]
        norms[row] = np.linalg.norm(weighted)
    query_norms = {}
    for query_id, counts in query_raw.items():
        query_norms[query_id] = math.sqrt(sum(
            (count * raw_idf[raw_vocabulary[word]]) ** 2
            for word, count in counts.items() if word in raw_vocabulary))
    if raw_pair_terms:
        numerator = np.bincount(np.asarray(raw_pair_index, dtype=np.int32),
            weights=np.asarray(raw_pair_tf) * np.asarray(raw_pair_qtf)
                    * raw_idf[np.asarray(raw_pair_terms, dtype=np.int32)] ** 2,
            minlength=pairs)
    else:
        numerator = np.zeros(pairs)
    denominators = norms[pair_rows] * np.asarray([query_norms[qid] for qid in query_ids])
    tfidf = np.divide(numerator, denominators, out=np.zeros(pairs), where=denominators > 0)
    metadata = {
        "documents": documents, "train_queries": len(qrels), "test_queries": len(test_queries),
        "train_pairs": training_pairs, "test_pairs": len(objects), "normalization": normalization,
        "raw_vocabulary_size": len(raw_vocabulary), "query_vocabulary_size": len(vocabulary),
        "field_token_totals": {"title": total_title, "body": total_body, "url": total_url},
        "feature_seconds": round(perf_counter() - started, 3),
        "tfidf_baseline": "raw tf, smoothed idf, exact full-document L2 cosine, default 2+ word token pattern",
    }
    return RankingData(query_ids=query_ids, doc_ids=doc_ids,
        labels=np.asarray(labels, dtype=np.int8), is_train=np.arange(pairs) < training_pairs,
        term_pair=term_pair, qtf=qtf, tf_title=tf_title, tf_body=tf_body, tf_url=tf_url,
        idf=idf[term_index], idf_url=idf_url[term_index], collection_prob=collection_prob[term_index],
        length_title=length_title, length_body=length_body, length_url=length_url,
        avg_title=total_title / documents, avg_body=total_body / documents, avg_url=total_url / documents,
        tfidf=tfidf, title_coverage=title_coverage, body_coverage=body_coverage,
        phrase_title=phrase_title, phrase_body=phrase_body, proximity=proximity, metadata=metadata)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_submission(path, objects, scores):
    path = Path(path)
    if len(objects) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite and cover every object exactly once")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
            dir=path.parent, prefix=".submission-", suffix=".csv") as output:
            temporary = Path(output.name)
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(["ObjectId", "Score"])
            for (object_id, _, _), value in zip(objects, scores, strict=True):
                writer.writerow([object_id, format(float(value), ".17g")])
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="HW2 classical text reranking")
    parser.add_argument("--submission_file", required=True, type=Path)
    parser.add_argument("--report_file", type=Path)
    parser.add_argument("--normalization", choices=["none", "lemma"], default="lemma")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    source_files = {args.data_dir.joinpath(name).resolve() for name in FILE_NAMES}
    if args.submission_file.resolve() in source_files:
        parser.error("Submission must not overwrite input data")
    if args.report_file and (args.report_file.resolve() in source_files
                            or args.report_file.resolve() == args.submission_file.resolve()):
        parser.error("Report must not overwrite input data or the submission")
    started = perf_counter()
    try:
        data = build_features(args.data_dir, args.normalization)
        config, report = select_parameters(data, seed=args.seed, holdout_fraction=0.3)
        predictions = score(data, config)
        objects = read_objects(args.data_dir / "objects.csv")
        write_submission(args.submission_file, objects, predictions[~data.is_train])
        report.update({"data": data.metadata, "normalization": args.normalization,
            "config": config, "seconds": round(perf_counter() - started, 3),
            "source_sha256": {name: file_hash(Path(__file__).parent / name)
                              for name in ("solution.py", "ranking.py", "requirements.txt")},
            "submission_sha256": file_hash(args.submission_file),
            "public_kaggle_score": None})
        if args.report_file:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            args.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"selected": config, "seconds": report["seconds"],
                          "submission": str(args.submission_file)}, ensure_ascii=False), flush=True)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
