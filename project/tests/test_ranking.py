import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranking import (NDCG, fit_component, load_component, normalize_scores,
                     predict_component, save_component, subset, transform)
from experiments import (audit_selection, dataset_hash, ensure_split, fit_final, make_split, paired_bootstrap,
                         select_from_tuning, split_manifest, train_and_select)


def toy_bundle(queries=24, candidates=6):
    rng = np.random.default_rng(81)
    qids = np.repeat([f"train:{q:04}" for q in range(queries)], candidates)
    test_ids = np.repeat(["test:0000", "test:0001"], candidates)
    X = rng.normal(size=(len(qids) + len(test_ids), 3)).astype(np.float32)
    labels = np.r_[np.tile([0, 0, 1, 1, 2, 3], queries), np.full(len(test_ids), -1)]
    X[:len(qids), 1] += labels[:len(qids)]
    return SimpleNamespace(X=X, labels=labels.astype(np.int8),
                           query_ids=np.r_[qids, test_ids],
                           doc_ids=np.array([f"doc-{n:04}" for n in range(len(X))]),
                           object_ids=np.r_[np.repeat("", len(qids)), np.arange(len(test_ids)).astype(str)],
                           is_train=labels >= 0,
                           feature_names=["bm25f_default", "bm25f_alternative", "coverage"],
                           metadata={})


class MetricTests(unittest.TestCase):
    def test_exp2_ties_equal_query_weight_and_unordered_rows(self):
        labels = np.array([0, 2, 0, 3, 1, 0])
        queries = np.array(["b", "a", "c", "a", "b", "a"])
        docs = np.array(["z", "a", "x", "b", "a", "c"])
        scores = np.array([3, 1, 4, 1, 2, 0])
        result = NDCG(labels, queries, docs, k=2)
        # a ties are document a before b, although b has the higher label.
        a = (3 + 7 / np.log2(3)) / (7 + 3 / np.log2(3))
        b = (0 + 1 / np.log2(3)) / 1
        np.testing.assert_allclose(result.per_query(scores), [a, b, 0])
        self.assertAlmostEqual(result.mean(scores), (a+b)/3)

    def test_metric_rejects_nonfinite_scores(self):
        metric = NDCG([1, 0], ["a", "a"])
        with self.assertRaises(ValueError):
            metric.mean([np.nan, 1])

    def test_group_local_ranks_ties_constants_singletons(self):
        X = np.array([[2, 7], [2, 7], [8, 7], [999, -1]], dtype=np.float32)
        qids = np.array(["a", "a", "a", "b"])
        ranked = transform(X, qids, "rank")
        np.testing.assert_allclose(ranked[:, 2:], [[.25, 0], [.25, 0], [1, 0], [0, 0]])
        X[-1] = [-1e6, 1e6]
        np.testing.assert_allclose(transform(X, qids, "rank")[:3], ranked[:3])
        normalized = normalize_scores([2, 2, 8, 90], qids, "zscore")
        self.assertEqual(normalized[-1], 0)
        self.assertAlmostEqual(normalized[:3].mean(), 0)
        self.assertAlmostEqual(normalized[:3].std(), 1)


class SplitAndSelectionTests(unittest.TestCase):
    def test_split_is_query_disjoint_and_fixed(self):
        qids = np.repeat(np.arange(14543).astype(str), 2)
        split = make_split(qids, 42)
        self.assertEqual([len(split[x]) for x in ("fit", "tune", "holdout")], [10543, 2000, 2000])
        self.assertFalse(set(split["fit"]) & set(split["tune"]))
        self.assertFalse(set(split["tune"]) & set(split["holdout"]))
        self.assertFalse(set(split["fit"]) & set(split["holdout"]))
        self.assertEqual(split, make_split(qids[::-1], 42))

    def test_saved_split_cannot_silently_change(self):
        bundle = toy_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            ensure_split(bundle, temporary, 42)
            with self.assertRaises(ValueError):
                ensure_split(bundle, temporary, 43)

    def test_holdout_labels_cannot_change_tuning_choice(self):
        bundle = toy_bundle()
        split = split_manifest(bundle)
        fit_ids = set(split["query_ids"]["fit"])
        tune_ids = set(split["query_ids"]["tune"])
        calls = []

        def fake_fit(train, config, validation=None, threads=4):
            self.assertEqual(set(train.query_ids), fit_ids)
            self.assertEqual(set(validation.query_ids), tune_ids)
            calls.append(config["name"])
            return object(), dict(config, iterations=2)

        def fake_predict(model, data, config, threads=4):
            column = config.get("column", "bm25f_alternative")
            return data.X[:, data.feature_names.index(column)].astype(np.float64)

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            with patch("experiments.fit_component", fake_fit), \
                    patch("experiments.predict_component", fake_predict), \
                    patch("experiments.save_component", return_value=None):
                initial = train_and_select(bundle, first, quick=True)
                mask = np.isin(bundle.query_ids, split["query_ids"]["holdout"])
                bundle.labels[mask] = 3 - bundle.labels[mask]
                again = train_and_select(bundle, second, quick=True)
            self.assertEqual(initial, again)
            self.assertEqual(len(calls), 8)
            self.assertFalse((Path(first) / "holdout.json").exists())
            saved = np.load(Path(first) / "tune-predictions.npz", allow_pickle=False)
            self.assertEqual(set(saved["query_ids"]), tune_ids)

    def test_changed_selection_is_rejected_after_audit_lock(self):
        bundle = toy_bundle()
        split = split_manifest(bundle)
        selection = {"format_version": 1, "seed": 42, "split_sha256": split["sha256"],
                     "input_feature_names": bundle.feature_names}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "audit-lock.json").write_text(json.dumps({"selection_sha256": "another"}))
            with self.assertRaisesRegex(ValueError, "frozen"):
                audit_selection(bundle, selection, directory)

    def test_bootstrap_is_paired(self):
        result = paired_bootstrap([.6, .8, .5], [.5, .7, .4], repeats=200)
        self.assertAlmostEqual(result["mean_difference"], .1)
        np.testing.assert_allclose(result["paired_bootstrap_95pct_ci"], [.1, .1])

    def test_final_refit_rejects_changed_fit_features_and_tune_labels(self):
        bundle = toy_bundle()
        split = split_manifest(bundle)
        fit = np.flatnonzero(np.isin(bundle.query_ids, split["query_ids"]["fit"]))
        tune = np.flatnonzero(np.isin(bundle.query_ids, split["query_ids"]["tune"]))
        selection = {"format_version": 1, "seed": 42, "split_sha256": split["sha256"],
                     "input_feature_names": bundle.feature_names,
                     "fit_sha256": dataset_hash(subset(bundle, fit)),
                     "tune_sha256": dataset_hash(subset(bundle, tune))}
        old = bundle.X[fit[0], 0].copy()
        bundle.X[fit[0], 0] = old + 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Fit features or labels"):
                fit_final(bundle, selection, directory)
            bundle.X[fit[0], 0] = old
            bundle.labels[tune[0]] = (bundle.labels[tune[0]] + 1) % 4
            with self.assertRaisesRegex(ValueError, "Tune features or labels"):
                fit_final(bundle, selection, directory)


class NativeModelTests(unittest.TestCase):
    def test_native_roundtrip_both_families_and_query_group_order(self):
        bundle = toy_bundle(queries=30)
        train = subset(bundle, np.flatnonzero(bundle.is_train))
        # Interleave rows so the fitter must group by QueryId before fitting.
        permutation = np.random.default_rng(17).permutation(len(train.labels))
        train = subset(train, permutation)
        for family in ("catboost", "lightgbm"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                config = {"name": "smoke", "family": family, "iterations": 8,
                          "seed": 42, "feature_mode": "rank", "columns": train.feature_names,
                          "params": {"min_data_in_leaf": 5} if family == "lightgbm" else {"depth": 3}}
                model, fitted = fit_component(train, config, threads=1)
                first = predict_component(model, bundle, fitted, threads=1)
                filename = save_component(model, fitted, directory, "ranker")
                loaded = load_component(directory, fitted, filename)
                np.testing.assert_array_equal(first, predict_component(loaded, bundle, fitted, threads=1))
                self.assertTrue(np.isfinite(first).all())
                self.assertGreater(np.ptp(first), 0)

    def test_train_validation_queries_cannot_overlap(self):
        train = subset(toy_bundle(), np.arange(12))
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_component(train, {"family": "catboost"}, train)


if __name__ == "__main__":
    unittest.main()
