#!/usr/bin/env python3
"""Reproducible stress data and exact labels; never a replacement for Kaggle data."""

import argparse
import csv
import json
from pathlib import Path
import random

VOCABULARY = 100_003
WORDS_PER_DOCUMENT = 80
INVERSE = pow(7919, -1, VOCABULARY)


def document_id(number):
    return str(1_000_000 + 17 * number)


def query_text(number, documents):
    kind = number % 8
    return ["ОБЩИЙ", f"общий группа{number % 200}", "чётный третий",
            f"уник{number * 37 % documents}_0", f"терм{number * 37 % VOCABULARY} общий",
            "zz_missing", "НОВОСТЬ новость общий", "!!!"][kind]


def expected_label(query, document, documents):
    # Derived from how documents are constructed, with no production code imports.
    kind = query % 8
    if kind in (0, 6, 7):
        return 1
    if kind == 1:
        return int(document % 200 == query % 200)
    if kind == 2:
        return int(document % 6 == 0)
    if kind == 3:
        return int(document == query * 37 % documents)
    if kind == 4:
        position = ((query * 37 % VOCABULARY - document * 97) * INVERSE) % VOCABULARY
        return int(position < WORDS_PER_DOCUMENT)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--documents", type=int, default=60_000)
    parser.add_argument("--queries", type=int, default=3_000)
    parser.add_argument("--objects", type=int, default=600_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.documents, args.queries, args.objects) <= 0:
        parser.error("all counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ["vkmarco-docs.tsv", "vkmarco-doceval-queries.tsv", "objects.csv",
             "sample_submission.csv", "expected_submission.csv", "synthetic.json"]
    if any((args.output_dir / name).exists() for name in names):
        parser.error("output already contains data; choose an empty directory")
    with (args.output_dir / names[0]).open("w", encoding="utf-8") as output:
        for number in range(args.documents):
            body = ["общий"]
            body += [f"терм{(number * 97 + j * 7919) % VOCABULARY}"
                     for j in range(WORDS_PER_DOCUMENT)]
            body += [f"уник{number}_{j}" for j in range(4)]
            if number % 2 == 0:
                body.append("чётный")
            if number % 3 == 0:
                body.append("третий")
            output.write(f"{document_id(number)}\thttps://urlonly.example/{number}\t"
                         f"Новость {document_id(number)} ГРУППА{number % 200}\t{' '.join(body)}\n")
    with (args.output_dir / names[1]).open("w", encoding="utf-8") as output:
        for number in range(args.queries):
            output.write(f"q{number}\t{query_text(number, args.documents)}\n")
    rng = random.Random(args.seed)
    positives = 0
    with (args.output_dir / names[2]).open("w", encoding="utf-8", newline="") as objects, \
            (args.output_dir / names[4]).open("w", encoding="utf-8", newline="") as expected:
        objects_writer = csv.writer(objects, lineterminator="\n")
        expected_writer = csv.writer(expected, lineterminator="\n")
        objects_writer.writerow(["ObjectId", "QueryId", "DocumentId"])
        expected_writer.writerow(["ObjectId", "Label"])
        for number in range(args.objects):
            query = rng.randrange(args.queries)
            document = rng.randrange(args.documents)
            if number % 7 == 0 and query % 8 == 3:
                document = query * 37 % args.documents
            label = expected_label(query, document, args.documents)
            positives += label
            object_id = f"obj{number:09}"
            objects_writer.writerow([object_id, f"q{query}", document_id(document)])
            expected_writer.writerow([object_id, label])
    with (args.output_dir / names[3]).open("w", encoding="utf-8", newline="") as sample:
        writer = csv.writer(sample, lineterminator="\n")
        writer.writerow(["ObjectId", "Label"])
        for number in reversed(range(args.objects)):
            writer.writerow([f"obj{number:09}", rng.randrange(2)])
    metadata = {"kind": "synthetic", "seed": args.seed, "documents": args.documents,
                "queries": args.queries, "objects": args.objects, "positives": positives,
                "description": "Synthetic stress data; not the course corpus or a Kaggle score."}
    (args.output_dir / names[5]).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
