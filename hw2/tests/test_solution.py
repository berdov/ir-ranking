"""Independent checks of ranking mathematics, input validation, and the CLI.

Run from hw2: .venv/bin/python -m unittest discover -s tests -v
No validation result on these artificial data is a competition score.
"""

from collections import Counter
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ranking import RankingData, ndcg, per_query_ndcg, score, select_parameters
from scripts.make_synthetic_data import FILES, generate_dataset
from solution import Normalizer, build_features, read_objects, read_qrels, read_queries


def scalar_fixture():
    """Two query terms and three candidates, including a zero-overlap document."""
    return RankingData(
        query_ids=["train:q"] * 3,
        doc_ids=["a", "b", "c"],
        labels=np.array([3, 1, 0], dtype=np.int8),
        is_train=np.ones(3, dtype=bool),
        term_pair=np.repeat(np.arange(3, dtype=np.int32), 2),
        qtf=np.array([2, 1] * 3, dtype=float),
        tf_title=np.array([2, 0, 0, 1, 0, 0], dtype=float),
        tf_body=np.array([1, 4, 3, 0, 0, 0], dtype=float),
        tf_url=np.array([0, 2, 1, 0, 0, 0], dtype=float),
        idf=np.array([0.4, 1.1] * 3),
        idf_url=np.array([0.2, 0.9] * 3),
        collection_prob=np.array([0.025, 0.05] * 3),
        length_title=np.array([4, 2, 0], dtype=float),
        length_body=np.array([10, 8, 20], dtype=float),
        length_url=np.array([3, 4, 0], dtype=float),
        avg_title=2.0,
        avg_body=12.0,
        avg_url=4.0,
        tfidf=np.array([0.8, 0.2, 0.0]),
        title_coverage=np.array([0.5, 0.5, 0]),
        body_coverage=np.array([1.0, 0.5, 0]),
        phrase_title=np.zeros(3),
        phrase_body=np.zeros(3),
        proximity=np.zeros(3),
    )


def small_corpus(destination):
    """A tiny reference corpus whose cosine similarity can be computed by hand."""
    destination = Path(destination)
    docs = {
        "a": ("", "alpha", "alpha beta gamma gamma"),
        "b": ("https://example.invalid/alpha", "beta", "beta delta"),
        "c": ("", "", ""),
        "d": ("", "alpha", "epsilon epsilon"),
        "unused": ("", "omega", "epsilon"),
    }
    with (destination / "vkmarco-docs.tsv").open("w", encoding="utf-8") as handle:
        for doc_id, fields in docs.items():
            handle.write("\t".join((doc_id,) + fields) + "\n")
    (destination / "vkmarco-doctrain-queries.tsv").write_text(
        "100\talpha\n200\tbeta\n", encoding="utf-8"
    )
    (destination / "vkmarco-doctrain-qrels.tsv").write_text(
        "100 0 a 3\n100 0 b 0\n200 0 a 1\n200 0 b 3\n", encoding="utf-8"
    )
    (destination / "vkmarco-doceval-queries.tsv").write_text(
        "100\tgamma gamma absent\n900\talpha\n950\t!!!\n", encoding="utf-8"
    )
    (destination / "objects.csv").write_text(
        "ObjectId,QueryId,DocumentId\n101,100,a\n102,100,b\n990,900,a\n994,900,d\n1001,950,c\n",
        encoding="utf-8",
    )
    (destination / "sample_submission.csv").write_text(
        "ObjectId,Score\n101,0\n102,0\n990,0\n994,0\n1001,0\n", encoding="utf-8"
    )
    return docs


def reference_cosine(documents, query, doc_id):
    """Unigram TF-IDF with smooth IDF and full vocabulary L2 normalization."""
    tokenize = lambda text: re.findall(r"(?u)\b\w\w+\b", text.lower())
    counts = {key: Counter(tokenize(title + " " + body))
              for key, (_, title, body) in documents.items()}
    frequencies = Counter(token for terms in counts.values() for token in terms)
    idfs = {term: math.log((1 + len(counts)) / (1 + df)) + 1
            for term, df in frequencies.items()}
    qcounts = Counter(tokenize(query))
    qweights = {term: tf * idfs[term] for term, tf in qcounts.items() if term in idfs}
    dweights = {term: tf * idfs[term] for term, tf in counts[doc_id].items()}
    qnorm = math.sqrt(sum(value * value for value in qweights.values()))
    dnorm = math.sqrt(sum(value * value for value in dweights.values()))
    if not qnorm or not dnorm:
        return 0.0
    return sum(weight * dweights.get(term, 0) for term, weight in qweights.items()) / (qnorm * dnorm)


class FormulaTests(unittest.TestCase):
    def test_bm25_matches_independent_scalar_formula(self):
        data = scalar_fixture()
        k1, b = 1.7, 0.65
        expected = np.zeros(3)
        for edge, pair in enumerate(data.term_pair):
            tf = data.tf_title[edge] + data.tf_body[edge]
            length = data.length_title[pair] + data.length_body[pair]
            norm = 1 - b + b * length / (data.avg_title + data.avg_body)
            expected[pair] += data.idf[edge] * (k1 + 1) * tf / (tf + k1 * norm)
        actual = score(data, {"model": "bm25", "k1": k1, "b": b})
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        # Repetition in a query is deliberately ignored by this BM25 variant.
        np.testing.assert_allclose(actual, score(replace(data, qtf=data.qtf * 7),
                                               {"model": "bm25", "k1": k1, "b": b}))

    def test_bm25f_combines_fields_before_saturation(self):
        data = scalar_fixture()
        config = {"model": "bm25f", "k1": 1.3, "b_title": 0.2, "b_body": 0.8,
                  "b_url": 0.35, "w_title": 2.5, "w_url": 0.6}
        expected = np.zeros(3)
        for edge, pair in enumerate(data.term_pair):
            combined = 0
            for field, weight in (("title", config["w_title"]), ("body", 1.0), ("url", config["w_url"])):
                tf = getattr(data, "tf_" + field)[edge]
                length = getattr(data, "length_" + field)[pair]
                average = getattr(data, "avg_" + field)
                b = config["b_" + field]
                combined += weight * tf / (1 - b + b * length / average)
            expected[pair] += data.idf_url[edge] * combined * (1 + config["k1"]) / (combined + config["k1"])
        np.testing.assert_allclose(score(data, config), expected, atol=1e-12)

    def test_dirichlet_accounts_for_missing_terms_and_length(self):
        data = scalar_fixture()
        mu = 50.0
        expected = np.zeros(3)
        for edge, pair in enumerate(data.term_pair):
            tf = data.tf_title[edge] + data.tf_body[edge]
            length = data.length_title[pair] + data.length_body[pair]
            probability = (tf + mu * data.collection_prob[edge]) / (length + mu)
            # The implementation removes the query-only constant log p_C.
            expected[pair] += data.qtf[edge] * math.log(probability / data.collection_prob[edge])
        np.testing.assert_allclose(score(data, {"model": "lm", "mu": mu}), expected, atol=1e-12)
        shorter = replace(data, length_body=np.array([10, 8, 1], dtype=float))
        self.assertGreater(score(shorter, {"model": "lm", "mu": mu})[2], expected[2])

    def test_oov_terms_and_empty_fields_produce_finite_scores(self):
        data = scalar_fixture()
        entirely_unknown = replace(data, collection_prob=np.zeros_like(data.collection_prob),
                                   tf_title=np.zeros_like(data.tf_title), tf_body=np.zeros_like(data.tf_body))
        np.testing.assert_array_equal(score(entirely_unknown, {"model": "lm", "mu": 1000}), np.zeros(3))
        empty = replace(entirely_unknown, tf_url=np.zeros_like(data.tf_url),
                        length_title=np.zeros(3), length_body=np.zeros(3), length_url=np.zeros(3),
                        avg_title=0, avg_body=0, avg_url=0)
        config = {"model": "bm25f", "b_title": 1, "b_body": 1, "b_url": 1, "w_url": 1}
        np.testing.assert_array_equal(score(empty, config), np.zeros(3))

    def test_ndcg_exponential_gain_and_query_average(self):
        labels = np.array([3, 2, 0, 1, 1])
        scores = np.array([1, 3, 2, 0, 1])
        query_ids = ["a", "a", "a", "b", "b"]
        doc_ids = ["0", "1", "2", "3", "4"]
        expected_a = (3 + 7 / math.log2(4)) / (7 + 3 / math.log2(3))
        self.assertAlmostEqual(ndcg(labels, scores, query_ids, doc_ids), (expected_a + 1) / 2)
        per_query = per_query_ndcg(labels, scores, query_ids, doc_ids)
        self.assertAlmostEqual(per_query["a"], expected_a)
        self.assertAlmostEqual(per_query["b"], 1)

    def test_ndcg_ties_use_document_id_and_zero_gain_is_zero(self):
        self.assertAlmostEqual(ndcg([0, 3], [1, 1], ["q", "q"], ["b", "a"]), 1)
        self.assertAlmostEqual(ndcg([3, 0], [1, 1], ["q", "q"], ["b", "a"]), 1 / math.log2(3))
        self.assertEqual(ndcg([0, 0], [1, 2], ["q", "q"], ["a", "b"]), 0)

    def test_ndcg_cutoff_ignores_eleventh_position(self):
        labels = [0] * 12
        labels[10] = 3
        scores = list(range(12, 0, -1))
        docs = [f"doc-{index:02d}" for index in range(12)]
        self.assertEqual(ndcg(labels, scores, ["q"] * 12, docs, k=10), 0)
        self.assertAlmostEqual(ndcg(labels, scores, ["q"] * 12, docs, k=11), 1 / math.log2(12))


class InputAndFeatureTests(unittest.TestCase):
    def test_raw_tfidf_exact_cosine_includes_nonquery_document_words(self):
        with tempfile.TemporaryDirectory() as directory:
            docs = small_corpus(directory)
            data = build_features(directory, normalization="none")
            query_text = {"train:100": "alpha", "train:200": "beta",
                          "test:100": "gamma gamma absent", "test:900": "alpha", "test:950": "!!!"}
            expected = [reference_cosine(docs, query_text[qid], did)
                        for qid, did in zip(data.query_ids, data.doc_ids)]
            np.testing.assert_allclose(data.tfidf, expected, atol=1e-12)
            np.testing.assert_allclose(score(data, {"model": "tfidf"}), expected, atol=1e-12)
            self.assertTrue(np.all(data.labels[~data.is_train] == -1))
            self.assertIn("train:100", data.query_ids)
            self.assertIn("test:100", data.query_ids)

    def test_normalization_handles_russian_inflection_and_empty_input(self):
        normalizer = Normalizer("lemma")
        self.assertEqual(normalizer.tokens("КОШКИ кошку"), ["кошка", "кошка"])
        self.assertEqual(normalizer.tokens("!!!"), [])
        self.assertEqual(normalizer.tokens(""), [])
        self.assertEqual(Normalizer("none").tokens("Ёжики"), ["ежики"])
        self.assertEqual(Normalizer("none").tokens("Кари\u0341бский"), ["карибский"])
        self.assertEqual(Normalizer("none").tokens("ЙОД май"), ["йод", "май"])

    def test_query_and_qrel_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.tsv"
            path.write_text("17\tтекст\n2001\tдругой текст\n", encoding="utf-8")
            self.assertEqual(read_queries(path), {"17": "текст", "2001": "другой текст"})
            path.write_text("17 0 doc-x 3\n17 0 doc-y 0\n", encoding="utf-8")
            self.assertEqual(read_qrels(path), {"17": {"doc-x": 3, "doc-y": 0}})
            for invalid in ("17 0 doc-x 4\n", "17 0 doc-x -1\n", "17 0 doc-x bad\n"):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_qrels(path)

    def test_object_ids_must_be_unique_and_header_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects.csv"
            for invalid in (
                "ObjectId,QueryId,DocumentId\n7,1,a\n7,1,b\n",
                "ObjectId,Label\n7,1\n",
            ):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_objects(path)

    def test_unknown_query_and_missing_document_are_rejected(self):
        for row in ("101,999999,a", "101,100,missing-document"):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                small_corpus(directory)
                (Path(directory) / "objects.csv").write_text(
                    "ObjectId,QueryId,DocumentId\n" + row + "\n", encoding="utf-8"
                )
                (Path(directory) / "sample_submission.csv").write_text(
                    "ObjectId,Score\n101,0\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "[Uu]nknown.*(?:QueryId|DocumentId)"):
                    build_features(directory, normalization="none")

    def test_old_homework_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "lenta-ru-news.csv").write_text("url,title,text\n", encoding="utf-8")
            (Path(directory) / "objects.csv").write_text("ObjectId,QueryId,DocumentId\n", encoding="utf-8")
            with self.assertRaises((FileNotFoundError, ValueError)):
                build_features(directory, normalization="none")


class PipelineTests(unittest.TestCase):
    def test_parameter_selection_uses_training_queries_only(self):
        with tempfile.TemporaryDirectory() as directory:
            generate_dataset(directory)
            data = build_features(directory, normalization="none")
            chosen, report = select_parameters(data, seed=42, holdout_fraction=0.3)
            splits = report["splits"]
            tune = set(splits["tune_query_ids"])
            holdout = set(splits["holdout_query_ids"])
            expected = {qid for qid, train in zip(data.query_ids, data.is_train) if train}
            self.assertTrue(tune and holdout)
            self.assertFalse(tune & holdout)
            self.assertEqual(tune | holdout, expected)
            self.assertTrue(all(qid.startswith("train:") for qid in tune | holdout))

            # Neither hidden test labels nor test feature magnitudes can select a model.
            poison_labels = data.labels.copy()
            poison_labels[~data.is_train] = np.arange((~data.is_train).sum()) % 4
            poison_tfidf = data.tfidf.copy()
            poison_tfidf[~data.is_train] = 12345
            poisoned = replace(data, labels=poison_labels, tfidf=poison_tfidf)
            try:
                chosen_again, report_again = select_parameters(poisoned, seed=42, holdout_fraction=0.3)
            except ValueError as error:
                self.assertIn("test", str(error).lower())
            else:
                self.assertEqual(chosen_again, chosen)
                self.assertEqual(report_again["splits"], report["splits"])
                self.assertEqual(report_again["candidates"], report["candidates"])

            # Holdout is evaluated only after choosing the complete formula.
            holdout_labels = data.labels.copy()
            holdout_rows = np.array([qid in holdout for qid in data.query_ids])
            holdout_labels[holdout_rows] = 3 - holdout_labels[holdout_rows]
            holdout_poisoned = replace(data, labels=holdout_labels)
            chosen_holdout, report_holdout = select_parameters(holdout_poisoned, seed=42, holdout_fraction=0.3)
            self.assertEqual(chosen_holdout, chosen)
            self.assertEqual(report_holdout["candidates"], report["candidates"])

    def test_cli_submission_is_finite_complete_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_dir = directory / "data"
            counts = generate_dataset(data_dir)
            self.assertEqual(set(path.name for path in data_dir.iterdir()), set(FILES))
            first = directory / "first.csv"
            second = directory / "second.csv"
            report = directory / "report.json"
            for destination in (first, second):
                command = [sys.executable, str(ROOT / "solution.py"),
                           f"--submission_file={destination}", f"--report_file={report}",
                           str(data_dir)]
                completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with first.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["ObjectId", "Score"])
                rows = list(reader)
            self.assertEqual(len(rows), counts["objects"])
            self.assertEqual(len({row["ObjectId"] for row in rows}), counts["objects"])
            self.assertTrue(all(math.isfinite(float(row["Score"])) for row in rows))
            expected_ids = [oid for oid, _, _ in read_objects(data_dir / "objects.csv")]
            self.assertEqual([row["ObjectId"] for row in rows], expected_ids)
            with report.open(encoding="utf-8") as handle:
                self.assertIsInstance(json.load(handle), dict)


if __name__ == "__main__":
    unittest.main()
