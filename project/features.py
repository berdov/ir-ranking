"""Streaming lexical features for VK MARCO candidate reranking.

Statistics use the supplied corpus, never relevance labels. Only query-term
counts are retained; there is no corpus-wide document/term matrix in memory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from time import perf_counter
from urllib.parse import unquote, urlsplit

import numpy as np
import Stemmer

from data import FILE_NAMES, FeatureBundle, file_hash, load_pairs


SCHEMA_VERSION = 1
WORDS = re.compile(r"(?u)\b\w\w+\b")
FIELD_NAMES = ("title", "body", "url")


class Normalizer:
    """Russian Snowball in libstemmer; its own bounded cache avoids Python maps."""

    def __init__(self):
        self.stemmer = Stemmer.Stemmer("russian")
        self.stemmer.maxCacheSize = 100000

    def raw_tokens(self, text):
        return WORDS.findall(text.lower().replace("ё", "е").replace("\u0301", "").replace("\u0300", ""))

    def stem(self, tokens):
        return self.stemmer.stemWords(tokens)

    def tokens(self, text):
        return self.stem(self.raw_tokens(text))


def url_parts(url):
    try:
        parts = urlsplit(url)
        hostname = unquote(parts.hostname or "")
        path, query = unquote(parts.path), unquote(parts.query)
        return hostname + " " + path + " " + query, (
            len(url), len(hostname), path.count("/"), len(query),
            int(parts.scheme.lower() == "https"), sum(char.isdigit() for char in url),
            int(bool(query)), len(path),
        )
    except ValueError:
        return unquote(url), (len(url), 0, url.count("/"), 0, 0,
                             sum(char.isdigit() for char in url), 0, 0)


def phrase_present(joined, phrase):
    return float(bool(phrase) and (" " + " ".join(phrase) + " ") in joined)


def proximity(positions, wanted, length):
    """Minimum covering window / first and last match, strictly within a field."""
    present = [positions.get(term, ()) for term in wanted]
    matches = [(position, term) for term, items in zip(wanted, present) for position in items]
    if not matches:
        return 0., 1., 1.
    matches.sort()
    first, last = matches[0][0] / max(1, length), matches[-1][0] / max(1, length)
    if len(wanted) < 2 or any(not items for items in present):
        return 0., first, last
    counts, covered, left, best = {}, 0, 0, math.inf
    for right_position, term in matches:
        counts[term] = counts.get(term, 0) + 1
        if counts[term] == 1:
            covered += 1
        while covered == len(wanted):
            left_position, left_term = matches[left]
            best = min(best, right_position - left_position + 1)
            counts[left_term] -= 1
            if counts[left_term] == 0:
                covered -= 1
            left += 1
    return len(wanted) / best, first, last


def source_fingerprint(data_dir):
    """Fast cache identity, with full corpus SHA recorded during extraction.

    The corpus is keyed by size/mtime_ns plus hashes of its beginning and end.
    This local cache is an optimization, not a portable input-authenticity proof.
    Small inputs and feature-generating source files are hashed completely.
    """
    data_dir = Path(data_dir)
    files = {}
    for name in FILE_NAMES:
        path = data_dir / name
        stat = path.stat()
        info = {"size": stat.st_size}
        if name == "vkmarco-docs.tsv":
            digest = hashlib.sha256()
            with path.open("rb") as source:
                digest.update(source.read(65536))
                source.seek(max(0, stat.st_size - 65536))
                digest.update(source.read(65536))
            info.update(mtime_ns=stat.st_mtime_ns, edge_sha256=digest.hexdigest())
        else:
            info["sha256"] = file_hash(path)
        files[name] = info
    return {"schema_version": SCHEMA_VERSION, "files": files,
            "source_sha256": {name: file_hash(Path(__file__).with_name(name))
                              for name in ("data.py", "features.py")},
            "stemmer_version": importlib.metadata.version("PyStemmer")}


def _cached_bundle(cache_dir, fingerprint):
    matrix, documents = cache_dir / "features.npz", cache_dir / "documents.sqlite"
    if not matrix.is_file() or not documents.is_file():
        return None
    try:
        bundle = FeatureBundle.load(matrix)
        if not bundle.metadata.get("complete") or bundle.metadata.get("fingerprint") != fingerprint:
            return None
        with sqlite3.connect(documents.as_uri() + "?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE name='fingerprint'").fetchone()
            complete = connection.execute("SELECT value FROM metadata WHERE name='complete'").fetchone()
        if row is None or json.loads(row[0]) != fingerprint or complete != ("true",):
            return None
        return bundle
    except (OSError, ValueError, KeyError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid feature cache at {cache_dir}; rebuild with --force") from exc


def _field_extra(raw_tokens, stem_tokens, stem_counts, wanted_for_doc):
    positions = defaultdict(list)
    for index, token in enumerate(stem_tokens):
        if token in wanted_for_doc:
            positions[token].append(index)
    return (set(raw_tokens), " " + " ".join(raw_tokens) + " ",
            " " + " ".join(stem_tokens) + " ", positions)


def _matrix(pairs, tf, lengths, df_fields, df_combined, df_all, cf_fields,
            totals, documents, term_pair, term_index, qtf, query_size,
            extras, extra_names):
    n = len(pairs.labels)
    names, columns = list(extra_names), [extras[:, i] for i in range(extras.shape[1])]

    def add(name, values):
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"Nonfinite or malformed feature: {name}")
        names.append(name)
        columns.append(values)

    def aggregate(values):
        return np.bincount(term_pair, weights=values, minlength=n)

    def safe_ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)

    averages = totals / documents
    idf_tb = np.log1p((documents - df_combined + .5) / (df_combined + .5))[term_index]
    idf_all = np.log1p((documents - df_all + .5) / (df_all + .5))[term_index]
    idf_fields = [np.log1p((documents - freq + .5) / (freq + .5))[term_index] for freq in df_fields]
    sum_idf = aggregate(idf_all)
    add("query_term_count", query_size)
    add("query_idf_sum", sum_idf)
    add("query_idf_mean", safe_ratio(sum_idf, query_size))
    add("query_oov_fraction", safe_ratio(aggregate((df_all[term_index] == 0).astype(float)), query_size))

    def norm(field, b):
        average = averages[field]
        relative = lengths[field] / average if average > 0 else np.zeros(n)
        return (1 - b + b * relative)[term_pair]

    def field_bm25(field, k1=1.2, b=.6):
        denominator = tf[field] + k1 * norm(field, b)
        return aggregate(idf_fields[field] * safe_ratio(tf[field] * (k1 + 1), denominator))

    for field, name in enumerate(FIELD_NAMES):
        counts = tf[field]
        coverage = aggregate((counts > 0).astype(float))
        add(f"{name}_length", lengths[field])
        add(f"{name}_length_relative", lengths[field] / max(averages[field], 1e-12))
        add(f"{name}_coverage", safe_ratio(coverage, query_size))
        add(f"{name}_all_terms", ((coverage == query_size) & (query_size > 0)).astype(float))
        add(f"{name}_tf_sum_log", np.log1p(aggregate(counts)))
        add(f"{name}_tf_idf", aggregate(np.log1p(counts) * idf_fields[field]))
        add(f"{name}_idf_coverage", safe_ratio(aggregate(idf_all * (counts > 0)), sum_idf))
        add(f"{name}_tf_density", safe_ratio(aggregate(counts), lengths[field]))
        add(f"bm25_{name}", field_bm25(field))
        add(f"bm25_{name}_high_k", field_bm25(field, k1=2.5, b=.3 if field != 1 else .9))
        minimum = np.full(n, np.inf)
        maximum = np.zeros(n)
        np.minimum.at(minimum, term_pair, counts)
        np.maximum.at(maximum, term_pair, counts)
        minimum[~np.isfinite(minimum)] = 0
        add(f"{name}_tf_min_log", np.log1p(minimum))
        add(f"{name}_tf_max_log", np.log1p(maximum))
        probability = cf_fields[field][term_index] / max(float(totals[field]), 1.)
        known = probability > 0
        for mu in (300., 1500.):
            gain = np.zeros(len(term_pair))
            gain[known] = qtf[known] * np.log1p(counts[known] / (mu * probability[known]))
            score = aggregate(gain) - aggregate(qtf * known) * np.log1p(lengths[field] / mu)
            add(f"lm_{name}_mu{int(mu)}", score)

    # Fixed alternatives are features and train-only baseline candidates.
    # Field frequencies are normalized before the one shared saturation.
    configs = [("default", 2., 4., .25, .6)]
    configs += [(f"k{k:g}_t{t:g}_u{u:g}_b{b:g}".replace(".", "p"), k, t, u, b)
                for k, t, u, b in (
                    (1.2, 2., .25, .6), (1.2, 4., .25, .6), (2., 2., .25, .6),
                    (2., 8., .25, .6), (2., 4., 0., .6), (2., 4., 1., .6),
                    (.8, 4., .25, .6), (3., 4., .25, .6), (2., 4., .25, .3),
                    (2., 4., .25, .9), (1.2, 1., 0., .75), (2., 8., 1., .3),
                )]
    for name, k1, title_weight, url_weight, body_b in configs:
        weighted = title_weight * safe_ratio(tf[0], norm(0, .3))
        weighted += safe_ratio(tf[1], norm(1, body_b))
        if url_weight:
            weighted += url_weight * safe_ratio(tf[2], norm(2, .3))
        idf = idf_all if url_weight else idf_tb
        add("bm25f_" + name, aggregate(idf * weighted * (k1 + 1) / (weighted + k1)))
    combined_tf = tf[0] + tf[1]
    combined_length = lengths[0] + lengths[1]
    combined_average = averages[0] + averages[1]
    for k1, b in ((1.2, .6), (2., .9)):
        denominator = combined_tf + k1 * (1 - b + b * combined_length[term_pair] / max(combined_average, 1e-12))
        add(f"bm25_combined_k{k1:g}_b{b:g}".replace(".", "p"),
            aggregate(idf_tb * safe_ratio(combined_tf * (k1 + 1), denominator)))
    add("combined_idf_coverage", safe_ratio(aggregate(idf_all * ((tf[0] + tf[1] + tf[2]) > 0)), sum_idf))
    add("body_title_length_ratio", safe_ratio(lengths[1], lengths[0] + 1))
    add("title_query_length_ratio", safe_ratio(lengths[0], query_size + 1))
    return np.column_stack(columns).astype(np.float32, copy=False), names


def build_features(data_dir, cache_dir, force=False, max_documents=None):
    """Return FeatureBundle and atomically cache only a completed full corpus run.

    ``max_documents`` is a profiling mode: returns found candidate rows only,
    writes files with a ``profile-N`` suffix, and never reuses them as full data.
    """
    started = perf_counter()
    data_dir, cache_dir = Path(data_dir).resolve(), Path(cache_dir).resolve()
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint(data_dir)
    if not force and max_documents is None:
        cached = _cached_bundle(cache_dir, fingerprint)
        if cached is not None:
            print(f"Loaded completed feature cache: {cached.X.shape}", flush=True)
            return cached
    pairs = load_pairs(data_dir)
    normalizer = Normalizer()
    query_raw = {qid: normalizer.raw_tokens(text) for qid, text in pairs.query_text.items()}
    query_stem = {qid: normalizer.stem(tokens) for qid, tokens in query_raw.items()}
    query_counts = {qid: Counter(tokens) for qid, tokens in query_stem.items()}
    query_terms = {qid: list(counts) for qid, counts in query_counts.items()}
    vocabulary = sorted({term for terms in query_terms.values() for term in terms})
    term_ids = {term: i for i, term in enumerate(vocabulary)}
    n, v = len(pairs.labels), len(vocabulary)
    query_size = np.asarray([len(query_terms[qid]) for qid in pairs.query_ids], dtype=np.int32)
    offsets = np.r_[0, np.cumsum(query_size, dtype=np.int64)]
    term_pair = np.repeat(np.arange(n, dtype=np.int32), query_size)
    term_index = np.empty(int(offsets[-1]), dtype=np.int32)
    qtf = np.empty(len(term_pair), dtype=np.float32)
    by_document = defaultdict(list)
    for index, (qid, did) in enumerate(zip(pairs.query_ids, pairs.doc_ids)):
        start, end = offsets[index:index + 2]
        term_index[start:end] = [term_ids[term] for term in query_terms[qid]]
        qtf[start:end] = list(query_counts[qid].values())
        by_document[str(did)].append(index)
    tf = np.zeros((3, len(term_pair)), dtype=np.float32)
    lengths = np.zeros((3, n), dtype=np.float32)
    df_fields = np.zeros((3, v), dtype=np.int64)
    cf_fields = np.zeros((3, v), dtype=np.int64)
    df_combined, df_all = np.zeros(v, dtype=np.int64), np.zeros(v, dtype=np.int64)
    totals = np.zeros(3, dtype=np.int64)
    field_extra_names = ("raw_coverage", "phrase", "raw_phrase", "bigram_coverage", "proximity", "first_match", "last_match")
    extra_names = [f"{field}_{name}" for field in FIELD_NAMES for name in field_extra_names]
    extra_names += ["url_char_length_log", "hostname_length_log", "url_path_depth", "url_query_length_log",
                    "url_https", "url_digit_count", "url_has_query", "url_path_length_log",
                    "title_char_length_log", "body_char_length_log", "query_raw_length",
                    "query_number_fraction", "title_exact_query", "title_query_char_overlap"]
    extras = np.zeros((n, len(extra_names)), dtype=np.float32)
    seen, found = set(), np.zeros(n, dtype=bool)
    full_hash = hashlib.sha256()
    documents = 0
    handle, temporary_name = tempfile.mkstemp(prefix=".documents-", suffix=".sqlite", dir=cache_dir)
    os.close(handle)
    temporary_db = Path(temporary_name)
    connection = sqlite3.connect(temporary_db)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE documents(doc_id TEXT PRIMARY KEY, url TEXT, title TEXT, body TEXT)")
    connection.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT)")
    try:
        with (data_dir / "vkmarco-docs.tsv").open("rb") as source:
            for raw_line in source:
                if max_documents is not None and documents >= max_documents:
                    break
                full_hash.update(raw_line)
                fields = raw_line.decode("utf-8").rstrip("\r\n").split("\t", 3)
                fields += [""] * (4 - len(fields))
                did, url, title, body = fields
                if not did or did in seen:
                    raise ValueError(f"Empty or duplicate document ID at line {documents + 1}: {did}")
                seen.add(did)
                documents += 1
                url_text, url_info = url_parts(url)
                raw_fields = [normalizer.raw_tokens(text) for text in (title, body, url_text)]
                stems = [normalizer.stem(tokens) for tokens in raw_fields]
                counts = list(map(Counter, stems))
                field_lengths = [len(tokens) for tokens in stems]
                totals += field_lengths
                relevant = (counts[0].keys() | counts[1].keys() | counts[2].keys()) & term_ids.keys()
                for term in relevant:
                    tid = term_ids[term]
                    df_all[tid] += 1
                    if term in counts[0] or term in counts[1]:
                        df_combined[tid] += 1
                    for field in range(3):
                        value = counts[field].get(term, 0)
                        if value:
                            df_fields[field, tid] += 1
                            cf_fields[field, tid] += value
                pair_indices = by_document.get(did)
                if pair_indices:
                    connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?)", (did, url, title, body))
                    wanted_doc = set().union(*(query_terms[pairs.query_ids[index]] for index in pair_indices))
                    field_info = [_field_extra(raw_fields[f], stems[f], counts[f], wanted_doc) for f in range(3)]
                    for index in pair_indices:
                        qid = pairs.query_ids[index]
                        terms, raw_query = query_terms[qid], query_raw[qid]
                        raw_unique = set(raw_query)
                        query_phrase = query_stem[qid]
                        start, end = offsets[index:index + 2]
                        values = []
                        for field in range(3):
                            tf[field, start:end] = [counts[field].get(term, 0) for term in terms]
                            lengths[field, index] = field_lengths[field]
                            raw_set, raw_joined, joined, positions = field_info[field]
                            window, first, last = proximity(positions, terms, field_lengths[field])
                            bigrams = list(zip(query_phrase, query_phrase[1:]))
                            bigram_score = sum(phrase_present(joined, pair) for pair in bigrams) / max(1, len(bigrams))
                            values += [len(raw_set & raw_unique) / max(1, len(raw_unique)),
                                       phrase_present(joined, query_phrase), phrase_present(raw_joined, raw_query),
                                       bigram_score, window, first, last]
                        ui = list(url_info)
                        for field in (0, 1, 3, 7):
                            ui[field] = math.log1p(ui[field])
                        query_characters = set("".join(raw_query))
                        title_characters = set("".join(raw_fields[0]))
                        values += ui + [math.log1p(len(title)), math.log1p(len(body)), len(raw_query),
                                        sum(token.isdigit() for token in raw_query) / max(1, len(raw_query)),
                                        float(bool(raw_query) and raw_fields[0] == raw_query),
                                        len(query_characters & title_characters) / max(1, len(query_characters))]
                        extras[index] = values
                        found[index] = True
                if documents % 1000 == 0:
                    connection.commit()
                if documents % 20000 == 0:
                    elapsed = perf_counter() - started
                    print(f"documents={documents:,} candidate_pairs={int(found.sum()):,}/{n:,} elapsed={elapsed:.1f}s docs/s={documents/elapsed:.1f}", flush=True)
        if not documents:
            raise ValueError("Empty document corpus")
        if max_documents is None and not found.all():
            missing = np.unique(pairs.doc_ids[~found])
            raise ValueError(f"Unknown candidate document IDs: {missing[:5].tolist()}")
        if source_fingerprint(data_dir) != fingerprint:
            raise ValueError("Source files changed while extracting features; cache was not published")
        X, names = _matrix(pairs, tf, lengths, df_fields, df_combined, df_all, cf_fields,
                           totals, documents, term_pair, term_index, qtf, query_size, extras, extra_names)
        complete = max_documents is None
        keep = np.ones(n, dtype=bool) if complete else found
        metadata = {"schema_version": SCHEMA_VERSION, "complete": complete, "fingerprint": fingerprint,
                    "documents": documents, "candidate_documents": int(connection.execute("SELECT count(*) FROM documents").fetchone()[0]),
                    "train_pairs": int(np.count_nonzero(pairs.is_train & keep)), "test_pairs": int(np.count_nonzero(~pairs.is_train & keep)),
                    "train_queries": len(np.unique(pairs.query_ids[pairs.is_train & keep])),
                    "test_queries": len(np.unique(pairs.query_ids[~pairs.is_train & keep])),
                    "query_vocabulary_size": v, "field_token_totals": dict(zip(FIELD_NAMES, totals.tolist())),
                    "field_average_lengths": dict(zip(FIELD_NAMES, (totals / documents).tolist())),
                    "corpus_sha256": full_hash.hexdigest() if complete else None,
                    "corpus_prefix_sha256": None if complete else full_hash.hexdigest(),
                    "feature_seconds": round(perf_counter() - started, 3),
                    "normalization": "Russian Snowball (PyStemmer), lower case, ё→е, remove combining stress, 2+ word tokens",
                    "labels_used_for_features": False}
        bundle = FeatureBundle(X[keep], pairs.labels[keep], pairs.query_ids[keep], pairs.doc_ids[keep],
                               pairs.object_ids[keep], pairs.is_train[keep], names, metadata).validate()
        connection.executemany("INSERT INTO metadata VALUES (?, ?)",
            [("fingerprint", json.dumps(fingerprint)), ("complete", json.dumps(complete)), ("corpus_sha256", json.dumps(metadata["corpus_sha256"]))])
        connection.commit()
        connection.close()
        suffix = "" if complete else f".profile-{max_documents}"
        os.replace(temporary_db, cache_dir / f"documents{suffix}.sqlite")
        bundle.save(cache_dir / f"features{suffix}.npz")
        print(f"Features completed: {bundle.X.shape}, elapsed={perf_counter()-started:.1f}s, complete={complete}", flush=True)
        return bundle
    finally:
        connection.close()
        temporary_db.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_documents", type=int)
    args = parser.parse_args()
    build_features(args.data_dir, args.cache_dir, force=args.force, max_documents=args.max_documents)


if __name__ == "__main__":
    main()
