"""Validated competition input and portable, non-pickle feature archives."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


FILE_NAMES = (
    "vkmarco-docs.tsv", "vkmarco-doctrain-queries.tsv",
    "vkmarco-doctrain-qrels.tsv", "vkmarco-doceval-queries.tsv",
    "objects.csv", "sample_submission.csv",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_queries(path):
    queries = {}
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2 or not fields[0] or not fields[1].strip():
                raise ValueError(f"{path}:{number}: expected nonempty QueryId and QueryText")
            key, text = fields
            if key in queries:
                raise ValueError(f"{path}: duplicate QueryId {key}")
            queries[key] = text
    return queries


def read_csv(path, expected_header):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        if next(reader, None) != expected_header:
            raise ValueError(f"{path}: expected header {','.join(expected_header)}")
        for number, row in enumerate(reader, 2):
            if len(row) != len(expected_header) or any(not value.strip() for value in row):
                raise ValueError(f"{path}:{number}: malformed CSV row")
            yield row


@dataclass
class PairTable:
    query_ids: np.ndarray
    doc_ids: np.ndarray
    object_ids: np.ndarray
    labels: np.ndarray
    is_train: np.ndarray
    query_text: dict[str, str]


def load_pairs(data_dir):
    """Train rows grouped by query/document; test rows preserve objects.csv order."""
    data_dir = Path(data_dir)
    for name in FILE_NAMES:
        if not (data_dir / name).is_file():
            raise ValueError(f"Missing competition file: {data_dir / name}")
    train_queries = read_queries(data_dir / FILE_NAMES[1])
    test_queries = read_queries(data_dir / FILE_NAMES[3])
    overlap = set(train_queries) & set(test_queries)
    if overlap:
        raise ValueError(f"Training/test QueryId overlap: {sorted(overlap)[:3]}")
    if not train_queries or not test_queries:
        raise ValueError("Training and test queries must both be nonempty")
    qrels = {}
    with (data_dir / FILE_NAMES[2]).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"qrels line {number}: expected QueryId 0 DocumentId Label")
            qid, unused, did, raw_label = fields
            if qid not in train_queries or unused != "0":
                raise ValueError(f"qrels line {number}: unknown query or invalid second column")
            try:
                label = int(raw_label)
            except ValueError as exc:
                raise ValueError(f"qrels line {number}: noninteger label") from exc
            if not 0 <= label <= 3:
                raise ValueError(f"qrels line {number}: label must be 0, 1, 2 or 3")
            key = (qid, did)
            if key in qrels:
                raise ValueError(f"Duplicate training query/document pair: {key}")
            qrels[key] = label
    if set(qid for qid, _ in qrels) != set(train_queries):
        raise ValueError("Every training query must have labelled candidates")
    query_ids, doc_ids, labels, object_ids = [], [], [], []
    for (qid, did), label in sorted(qrels.items()):
        query_ids.append("train:" + qid)
        doc_ids.append(did)
        labels.append(label)
        object_ids.append("")
    n_train = len(labels)
    seen_objects, seen_pairs, seen_queries = set(), set(), set()
    for oid, qid, did in read_csv(data_dir / "objects.csv", ["ObjectId", "QueryId", "DocumentId"]):
        if oid in seen_objects or (qid, did) in seen_pairs:
            raise ValueError(f"Duplicate test object or query/document pair: {oid}, {qid}, {did}")
        if qid not in test_queries:
            raise ValueError(f"Unknown test QueryId: {qid}")
        seen_objects.add(oid)
        seen_pairs.add((qid, did))
        seen_queries.add(qid)
        query_ids.append("test:" + qid)
        doc_ids.append(did)
        labels.append(-1)
        object_ids.append(oid)
    if seen_queries != set(test_queries):
        raise ValueError("Every test query must have candidate objects")
    sample_ids = [row[0] for row in read_csv(data_dir / "sample_submission.csv", ["ObjectId", "Score"])]
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != seen_objects:
        raise ValueError("Sample submission IDs do not match objects.csv")
    query_text = {"train:" + qid: text for qid, text in train_queries.items()}
    query_text.update({"test:" + qid: text for qid, text in test_queries.items()})
    return PairTable(np.asarray(query_ids, dtype=str), np.asarray(doc_ids, dtype=str),
                     np.asarray(object_ids, dtype=str), np.asarray(labels, dtype=np.int8),
                     np.arange(len(labels)) < n_train, query_text)


@dataclass
class FeatureBundle:
    X: np.ndarray
    labels: np.ndarray
    query_ids: np.ndarray
    doc_ids: np.ndarray
    object_ids: np.ndarray
    is_train: np.ndarray
    feature_names: list[str]
    metadata: dict

    def validate(self):
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names):
            raise ValueError("Feature matrix and feature names do not match")
        size = len(self.X)
        if any(value.ndim != 1 or len(value) != size for value in
               (self.labels, self.query_ids, self.doc_ids, self.object_ids, self.is_train)):
            raise ValueError("Feature bundle row arrays have inconsistent shapes")
        if len(set(self.feature_names)) != len(self.feature_names) or not np.isfinite(self.X).all():
            raise ValueError("Feature names must be unique and all features finite")
        if self.is_train.dtype != np.dtype(bool):
            raise ValueError("is_train must be boolean")
        if np.any((self.labels[self.is_train] < 0) | (self.labels[self.is_train] > 3)):
            raise ValueError("Invalid training label")
        if np.any(self.labels[~self.is_train] != -1):
            raise ValueError("Test labels must be absent (represented by -1)")
        if np.any(self.object_ids[self.is_train] != ""):
            raise ValueError("Training rows must not contain ObjectIds")
        if len(np.unique(self.object_ids[~self.is_train])) != np.count_nonzero(~self.is_train):
            raise ValueError("Duplicate test ObjectId in feature bundle")
        if set(self.query_ids[self.is_train]) & set(self.query_ids[~self.is_train]):
            raise ValueError("Training and test groups overlap")
        return self

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".features-", suffix=".npz", delete=False) as output:
                temporary = Path(output.name)
                np.savez_compressed(output, X=self.X.astype(np.float32, copy=False),
                    labels=self.labels, query_ids=self.query_ids.astype(str),
                    doc_ids=self.doc_ids.astype(str), object_ids=self.object_ids.astype(str),
                    is_train=self.is_train, feature_names=np.asarray(self.feature_names, dtype=str),
                    metadata=np.asarray(json.dumps(self.metadata, ensure_ascii=False)))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            expected = {"X", "labels", "query_ids", "doc_ids", "object_ids", "is_train", "feature_names", "metadata"}
            if set(archive.files) != expected:
                raise ValueError("Unsupported feature archive schema")
            result = cls(X=archive["X"], labels=archive["labels"], query_ids=archive["query_ids"],
                doc_ids=archive["doc_ids"], object_ids=archive["object_ids"], is_train=archive["is_train"],
                feature_names=archive["feature_names"].tolist(), metadata=json.loads(str(archive["metadata"])))
        return result.validate()
