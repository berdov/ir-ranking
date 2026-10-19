#!/usr/bin/env python3
"""Validate a submission and optionally compare macOS/Linux scores and ordering."""

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path, header):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != header:
            raise ValueError(f"{path}: expected header {','.join(header)}")
        result = list(reader)
    if any(len(row) != len(header) or any(not value for value in row) for row in result):
        raise ValueError(f"{path}: empty or malformed row")
    if len({row[0] for row in result}) != len(result):
        raise ValueError(f"{path}: duplicate ObjectId")
    return result


def scores(path, expected):
    records = rows(path, ["ObjectId", "Score"])
    if [row[0] for row in records] != expected:
        raise ValueError(f"{path}: ObjectIds must match objects.csv completely and in its order")
    result = [float(row[1]) for row in records]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{path}: every Score must be finite")
    return result


def validate(data_dir, submission, comparison=None):
    objects = rows(Path(data_dir) / "objects.csv", ["ObjectId", "QueryId", "DocumentId"])
    if not objects:
        raise ValueError("objects.csv contains no candidates")
    expected = [row[0] for row in objects]
    sample = rows(Path(data_dir) / "sample_submission.csv", ["ObjectId", "Score"])
    if {row[0] for row in sample} != set(expected):
        raise ValueError("sample_submission.csv ObjectIds differ from objects.csv")
    primary = scores(submission, expected)
    result = {"passed": True, "submission": str(submission), "objects": len(objects),
              "queries": len({row[1] for row in objects}), "finite_scores": True,
              "object_ids_unique_complete_and_in_order": True}
    if comparison is not None:
        secondary = scores(comparison, expected)
        groups = {}
        for index, (object_id, query_id, _) in enumerate(objects):
            groups.setdefault(query_id, []).append((index, int(object_id)))
        different_queries = []
        for query_id, entries in groups.items():
            rank = lambda values: [object_id for _, object_id in sorted(
                entries, key=lambda pair: (-values[pair[0]], pair[1]))]
            if rank(primary) != rank(secondary):
                different_queries.append(query_id)
        close = all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                    for a, b in zip(primary, secondary))
        result["comparison"] = {"file": str(comparison), "scores_exactly_equal": primary == secondary,
            "scores_close": close, "relative_tolerance": 1e-12, "absolute_tolerance": 1e-12,
            "max_absolute_difference": max((abs(a - b) for a, b in zip(primary, secondary)), default=0),
            "rankings_equal": not different_queries, "different_query_ids": different_queries,
            "tie_break": "ascending numeric ObjectId"}
        result["passed"] = close and not different_queries
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--report_json", type=Path, required=True)
    args = parser.parse_args()
    protected = [args.submission, args.data_dir / "objects.csv", args.data_dir / "sample_submission.csv"]
    if args.compare:
        protected.append(args.compare)
    if args.report_json.resolve() in {path.resolve() for path in protected}:
        parser.error("The report must not overwrite an input file")
    try:
        result = validate(args.data_dir, args.submission, args.compare)
    except (OSError, ValueError) as error:
        result = {"passed": False, "error": str(error)}
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.report_json.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
