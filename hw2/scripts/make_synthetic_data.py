#!/usr/bin/env python3
"""Create a small artificial corpus for software checks, never for Kaggle scoring."""

import argparse
import csv
import json
from pathlib import Path


FILES = (
    "vkmarco-docs.tsv",
    "vkmarco-doctrain-queries.tsv",
    "vkmarco-doctrain-qrels.tsv",
    "vkmarco-doceval-queries.tsv",
    "objects.csv",
    "sample_submission.csv",
)

TOPICS = (
    ("кошка коты", "Кошки и коты", "Кошка живёт дома. Коты любят играть."),
    ("полет космос", "Полёт в космос", "Полет в космос и подготовка космонавтов."),
    ("чай зеленый", "Зелёный чай", "Зеленый чай заваривают горячей водой."),
    ("ремонт велосипеда", "Ремонт велосипеда", "Ремонт велосипеда: замена колеса."),
    ("router settings", "Router settings", "Router settings and internet connection."),
    ("поиск поиск текста", "Поиск текста", "Поиск текста. Поиск нужных слов."),
)


def generate_dataset(destination, train_queries=24, test_queries=6, force=False):
    """Write all six official-format files, with reproducible, noncontiguous IDs."""
    destination = Path(destination)
    if train_queries < 2 or test_queries < 1:
        raise ValueError("Need at least two training queries and one test query")
    destination.mkdir(parents=True, exist_ok=True)
    existing = [name for name in FILES if (destination / name).exists()]
    if existing and not force:
        raise FileExistsError("Refusing to overwrite existing data: " + ", ".join(existing))

    documents = []
    training = []
    testing = []
    qrels = []
    objects = []
    for split, count in (("train", train_queries), ("test", test_queries)):
        for index in range(count):
            # A QueryId may independently occur in both train and test files.
            query_id = str(100 + index * (37 if split == "train" else 53))
            query, title, body = TOPICS[(index + (1 if split == "test" else 0)) % len(TOPICS)]
            if split == "test" and index % 6 == 3:
                query = "несуществослово"
            if split == "test" and index % 6 == 4:
                query = "!!!"
            (training if split == "train" else testing).append((query_id, query))
            first_word = query.split()[0]
            url_keyword = "router" if "router" in query else "search"
            variants = (
                (f"https://example.invalid/{split}/{index}/guide", title, body),
                (f"https://example.invalid/{split}/{index}/notes", "Справочная страница", body),
                (f"https://example.invalid/{url_keyword}/{split}/{index}", first_word, "Погода сегодня солнечная."),
                ("", "", ""),
            )
            # Rotate field-only cases so all four labels do not encode one layout.
            if index % 4 == 1:
                variants = (variants[0][:2] + ("",),) + variants[1:]
            if split == "test" and index % 6 == 1:
                query = "router"
                testing[-1] = (query_id, query)
                variants = (
                    ("https://example.invalid/router/setup", "Настройка", "Инструкция подключения."),
                    ("https://example.invalid/setup", "Router", ""),
                    ("https://example.invalid/connect", "", "router router settings"),
                    ("", "", ""),
                )
            for candidate, (url, doc_title, doc_body) in enumerate(variants):
                doc_id = f"{split}-doc-{index * 101 + candidate * 13 + 7}"
                documents.append((doc_id, url, doc_title, doc_body))
                if split == "train":
                    qrels.append((query_id, "0", doc_id, str(3 - candidate)))
                else:
                    object_id = str(1009 + len(objects) * 29)
                    objects.append((object_id, query_id, doc_id))

    # Extra corpus words must contribute to document norms and corpus statistics.
    documents.extend((
        ("corpus-extra-901", "", "Внезапная статья", "яблоко груша банан"),
        ("corpus-extra-907", "", "", ""),
    ))
    for filename, rows in (
        ("vkmarco-docs.tsv", documents),
        ("vkmarco-doctrain-queries.tsv", training),
        ("vkmarco-doceval-queries.tsv", testing),
    ):
        with (destination / filename).open("w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write("\t".join(row) + "\n")
    with (destination / "vkmarco-doctrain-qrels.tsv").open("w", encoding="utf-8") as handle:
        for row in qrels:
            handle.write(" ".join(row) + "\n")
    with (destination / "objects.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ObjectId", "QueryId", "DocumentId"))
        writer.writerows(objects)
    with (destination / "sample_submission.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ObjectId", "Score"))
        writer.writerows((object_id, "0") for object_id, _, _ in objects)
    return {
        "synthetic": True,
        "train_queries": len(training),
        "test_queries": len(testing),
        "documents": len(documents),
        "objects": len(objects),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--train-queries", type=int, default=24)
    parser.add_argument("--test-queries", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Replace existing synthetic fixture files")
    args = parser.parse_args()
    counts = generate_dataset(args.destination, args.train_queries, args.test_queries, args.force)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("Synthetic fixture only: these scores cannot establish a Kaggle result.")


if __name__ == "__main__":
    main()
