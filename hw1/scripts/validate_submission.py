#!/usr/bin/env python3
"""Check every prediction directly against the source text, without the index.

This independent oracle uses only the Python standard library. Its memory is
outside the submission-generation budget: it keeps pairs and query terms in
memory, but reads the document corpus one physical line at a time.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


WORDS = re.compile(r"\w+")


def terms(text):
    # Tokenize first, then lowercase each token, as required by the template.
    # Deliberately no stemming, HTML stripping, URL terms, or ё -> е conversion.
    return {word.lower() for word in WORDS.findall(text)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path, header):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != header:
            raise ValueError(f"{path.name}: expected header {header!r}")
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"{path.name}:{number}: wrong column count")
            if any(not value for value in row):
                raise ValueError(f"{path.name}:{number}: empty value")
            yield number, row


def validate(data_dir, submission_file, report):
    paths = {name: data_dir / name for name in (
        "vkmarco-docs.tsv", "vkmarco-doceval-queries.tsv", "objects.csv",
        "sample_submission.csv",
    )}
    paths["submission"] = submission_file
    report["files"] = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
               "sha256": sha256(path)}
        for name, path in paths.items()
    }

    queries = {}
    with paths["vkmarco-doceval-queries.tsv"].open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f"queries:{number}: expected QueryId and text")
            query_id, text = fields
            if query_id in queries:
                raise ValueError(f"queries:{number}: duplicate QueryId {query_id}")
            queries[query_id] = terms(text)

    pairs = []
    object_ids = set()
    pairs_by_document = defaultdict(list)
    for number, row in csv_rows(paths["objects.csv"],
                                ["ObjectId", "QueryId", "DocumentId"]):
        object_id, query_id, document_id = row
        if object_id in object_ids:
            raise ValueError(f"objects.csv:{number}: duplicate ObjectId {object_id}")
        if query_id not in queries:
            raise ValueError(f"objects.csv:{number}: unknown QueryId {query_id}")
        object_ids.add(object_id)
        pairs_by_document[document_id].append((len(pairs), query_id))
        pairs.append((object_id, query_id, document_id))

    sample_ids = set()
    for number, (object_id, label) in csv_rows(paths["sample_submission.csv"],
                                              ["ObjectId", "Label"]):
        if object_id in sample_ids:
            raise ValueError(f"sample_submission.csv:{number}: duplicate ObjectId")
        if label not in {"0", "1"}:
            raise ValueError(f"sample_submission.csv:{number}: invalid Label")
        sample_ids.add(object_id)
    if sample_ids != object_ids:
        raise ValueError("sample ObjectIds differ from objects.csv: "
                         f"{len(object_ids - sample_ids)} missing, "
                         f"{len(sample_ids - object_ids)} extra")

    expected = bytearray(len(pairs))
    seen_documents = set()
    with paths["vkmarco-docs.tsv"].open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip("\r\n").split("\t", 3)
            if len(fields) != 4 or not fields[0]:
                raise ValueError(f"docs:{number}: expected Id, URL, Title, Body")
            document_id, _url, title, body = fields
            if document_id in seen_documents:
                raise ValueError(f"docs:{number}: duplicate DocumentId {document_id}")
            seen_documents.add(document_id)
            if document_id in pairs_by_document:
                document_terms = terms(title + " " + body)
                for pair_index, query_id in pairs_by_document[document_id]:
                    expected[pair_index] = queries[query_id].issubset(document_terms)
    missing_documents = pairs_by_document.keys() - seen_documents
    if missing_documents:
        raise ValueError(f"{len(missing_documents)} referenced documents are missing")

    mismatch_count = 0
    mismatch_examples = []
    order_errors = 0
    invalid_labels = 0
    duplicate_ids = 0
    submission_ids = set()
    submission_rows = 0
    submission_positives = 0
    for number, (object_id, label) in csv_rows(submission_file, ["ObjectId", "Label"]):
        position = submission_rows
        submission_rows += 1
        duplicate_ids += object_id in submission_ids
        submission_ids.add(object_id)
        if label not in {"0", "1"}:
            invalid_labels += 1
        else:
            submission_positives += int(label)
        if position >= len(pairs) or object_id != pairs[position][0]:
            order_errors += 1
        if position < len(pairs) and label != str(expected[position]):
            mismatch_count += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append({
                    "csv_line": number, "ObjectId": object_id,
                    "expected_ObjectId": pairs[position][0],
                    "QueryId": pairs[position][1], "DocumentId": pairs[position][2],
                    "expected_Label": expected[position], "actual_Label": label,
                })

    report["counts"] = {
        "documents": len(seen_documents), "queries": len(queries),
        "pairs": len(pairs), "referenced_documents": len(pairs_by_document),
        "sample_rows": len(sample_ids), "submission_rows": submission_rows,
        "expected_positives": sum(expected), "submission_positives": submission_positives,
        "expected_negatives": len(pairs) - sum(expected),
    }
    report["checks"] = {
        "row_count_matches": submission_rows == len(pairs),
        "object_id_set_matches": submission_ids == object_ids,
        "object_id_order_matches_objects_csv": order_errors == 0,
        "sample_object_id_set_matches_ignoring_order": True,
        "duplicate_submission_ids": duplicate_ids,
        "invalid_labels": invalid_labels,
        "object_id_order_errors": order_errors,
    }
    report["mismatches"] = mismatch_count
    report["mismatch_examples"] = mismatch_examples
    report["valid"] = (
        submission_rows == len(pairs) and submission_ids == object_ids
        and order_errors == 0 and duplicate_ids == 0 and invalid_labels == 0
        and mismatch_count == 0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--submission_file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "valid": False,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Direct per-document set containment over Title + Body; "
                  "Python re \\w+ then per-token lowercase; no production imports or index",
        "scope": "All provided pairs; independent local oracle, not a Kaggle score",
    }
    try:
        validate(args.data_dir, args.submission_file, report)
    except (OSError, ValueError, csv.Error) as error:
        report["error"] = str(error)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"},
                     ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
