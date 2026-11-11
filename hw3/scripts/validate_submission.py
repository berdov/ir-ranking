#!/usr/bin/env python3
"""Validate a HW3 submission against test.csv and the optional sample CSV."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_ids(path, *, submission=False):
    """Read identifiers exactly, without converting them to floating point."""
    ids, scores = [], []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if not header or len(set(header)) != len(header):
            raise ValueError(f"{Path(path).name}: missing or duplicate CSV header")
        if submission and header != ["ObjectId", "Score"]:
            raise ValueError(f"{Path(path).name}: expected exactly ObjectId,Score")
        if "ObjectId" not in header:
            raise ValueError(f"{Path(path).name}: missing ObjectId")
        id_column = header.index("ObjectId")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"{Path(path).name}: malformed row {reader.line_num}")
            identifier = row[id_column]
            if not identifier or identifier != identifier.strip():
                raise ValueError(f"{Path(path).name}: blank or padded ObjectId")
            ids.append(identifier)
            if submission:
                try:
                    score = float(row[1])
                except ValueError as error:
                    raise ValueError(f"{Path(path).name}: nonnumeric Score") from error
                if not math.isfinite(score):
                    raise ValueError(f"{Path(path).name}: non-finite Score")
                scores.append(score)
    if not ids:
        raise ValueError(f"{Path(path).name}: no objects")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{Path(path).name}: duplicate ObjectId")
    return ids, scores


def validate(data_dir, submission):
    data_dir, submission = Path(data_dir), Path(submission)
    expected, _ = read_ids(data_dir / "test.csv")
    actual, scores = read_ids(submission, submission=True)
    if actual != expected:
        if set(actual) != set(expected):
            raise ValueError("Submission ObjectId set differs from test.csv")
        raise ValueError("Submission row order differs from test.csv")
    sample = data_dir / "sample_submission.csv"
    sample_present = sample.is_file()
    if sample_present:
        sample_ids, _ = read_ids(sample, submission=True)
        if sample_ids != expected:
            raise ValueError("sample_submission.csv ObjectId/order differs from test.csv")
    return {
        "passed": True, "rows": len(actual), "unique_object_ids": len(set(actual)),
        "matches_test_order": True, "sample_checked": sample_present,
        "scores_finite": True, "min_score": min(scores), "max_score": max(scores),
        "submission_sha256": digest(submission),
        "test_sha256": digest(data_dir / "test.csv"),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--report_file", type=Path)
    args = parser.parse_args()
    if args.report_file:
        protected = {(args.data_dir / name).resolve() for name in
                     ("train.csv", "test.csv", "sample_submission.csv", "features.md")}
        protected.add(args.submission.resolve())
        if args.report_file.resolve() in protected:
            parser.error("Validation report must not overwrite an input file")
    try:
        result = validate(args.data_dir, args.submission)
    except (OSError, ValueError, csv.Error) as error:
        result = {"passed": False, "error": str(error)}
    if args.report_file:
        write_json(args.report_file, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
