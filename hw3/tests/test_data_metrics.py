"""Small independent fixtures for the CSV contract and ranking metric."""

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import Dataset, group_sizes, grouped, load_test, load_train, split_by_query
from metrics import mean_ndcg, per_query_ndcg


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, text, name="fixture.csv"):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_training_schema_float32_and_missing_numeric(self):
        dataset = load_train(self.write(
            "QueryId,Label,f1,f2\n001,3,1.5,\nNA,0,-2,NaN\n001,1,0,4\n"
        ))
        self.assertEqual(dataset.feature_names, ("f1", "f2"))
        self.assertEqual(dataset.X.dtype, np.float32)
        self.assertEqual(dataset.query_ids.tolist(), ["001", "NA", "001"])
        self.assertEqual(dataset.labels.tolist(), [3, 0, 1])
        self.assertTrue(np.isnan(dataset.X[:2, 1]).all())
        self.assertIsNone(dataset.object_ids)

    def test_test_loader_keeps_exact_object_id_and_does_not_need_train(self):
        dataset = load_test(self.write(
            "ObjectId,QueryId,f1\n00001,q2,1.5\n9007199254740993,q1,2\nNA,q2,3\n"
        ), ("f1",))
        self.assertEqual(dataset.object_ids.tolist(), ["00001", "9007199254740993", "NA"])
        self.assertIsNone(dataset.labels)
        self.assertEqual(dataset.feature_names, ("f1",))

    def test_malformed_training_csv_rejected(self):
        cases = [
            "", "QueryId,Label,f\n",
            "QueryId,Label,f,f\nq,1,2,3\n",
            "QueryId,Label,\nq,1,2\n",
            "QueryId,Label, f\nq,1,2\n",
            "QueryId,Label\nq,1\n",
            "QueryId,f\nq,1\n",
            "QueryId,Label,ObjectId,f\nq,1,100,2\n",
            "QueryId,Label,f\nq,1,2,3\n",
            "QueryId,Label,f\nq,1\n",
            "QueryId,Label,f\nq,1,word\n",
            "QueryId,Label,f\nq,1,inf\n",
            "QueryId,Label,f\nq,1,-inf\n",
            "QueryId,Label,f\nq,1,1e50\n",
            "QueryId,Label,f\nq,-1,2\n",
            "QueryId,Label,f\nq,4,2\n",
            "QueryId,Label,f\nq,1.5,2\n",
            "QueryId,Label,f\nq,,2\n",
            "QueryId,Label,f\nq,nan,2\n",
            "QueryId,Label,f\nq,inf,2\n",
            "QueryId,Label,f\n,1,2\n",
            "QueryId,Label,f\n q,1,2\n",
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_train(self.write(text))

    def test_test_feature_mismatch_duplicates_and_label_leakage_rejected(self):
        cases = [
            "ObjectId,QueryId,f2,f1\n1,q,2,3\n",
            "ObjectId,QueryId,f1\n1,q,2\n",
            "ObjectId,QueryId,Label,f1,f2\n1,q,3,2,3\n",
            "ObjectId,QueryId,f1,f2\n1,q,2,3\n1,r,4,5\n",
            "ObjectId,QueryId,f1,f2\n,q,2,3\n",
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_test(self.write(text), ("f1", "f2"))

    def test_stable_group_sort_and_restore(self):
        original = Dataset(
            np.arange(5, dtype=np.float32)[:, None],
            np.array(["q2", "q1", "q2", "q3", "q1"]),
            np.array([3, 0, 1, 2, 3]), np.array(["a", "b", "c", "d", "e"]), ("f",),
        )
        ordered, permutation = grouped(original)
        np.testing.assert_array_equal(permutation, [1, 4, 0, 2, 3])
        np.testing.assert_array_equal(group_sizes(ordered.query_ids), [2, 2, 1])
        restored = np.empty(5)
        restored[permutation] = ordered.X[:, 0]
        np.testing.assert_array_equal(restored, original.X[:, 0])
        self.assertEqual(ordered.object_ids.tolist(), ["b", "e", "a", "c", "d"])
        with self.assertRaises(ValueError):
            group_sizes(original.query_ids)

    def test_split_is_query_disjoint_deterministic_and_row_order_independent(self):
        ids = np.array([f"q{i}" for i in range(20) for _ in range(3)])
        train, valid = split_by_query(ids, .2, 42)
        train2, valid2 = split_by_query(ids, .2, 42)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(valid, valid2)
        self.assertFalse(set(ids[train]) & set(ids[valid]))
        self.assertEqual(len(set(ids[valid])), 4)
        np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.arange(len(ids)))
        reverse = ids[::-1]
        _, reverse_valid = split_by_query(reverse, .2, 42)
        self.assertEqual(set(ids[valid]), set(reverse[reverse_valid]))

    def test_split_rejects_impossible_holdouts(self):
        for ids, fraction in [(["q"], .2), ([], .2), (["q", "r"], 0), (["q", "r"], 1)]:
            with self.subTest(ids=ids, fraction=fraction), self.assertRaises(ValueError):
                split_by_query(np.array(ids), fraction)


class MetricTests(unittest.TestCase):
    def test_hand_computed_ndcg_and_equal_query_weight(self):
        # q1 predicts relevance 0 then 3; q2 is a perfect single result.
        expected_q1 = 1 / math.log2(3)
        qids, values = per_query_ndcg([3, 0, 2], [0, 1, 5], ["q1", "q1", "q2"])
        self.assertEqual(qids.tolist(), ["q1", "q2"])
        np.testing.assert_allclose(values, [expected_q1, 1])
        self.assertAlmostEqual(mean_ndcg([3, 0, 2], [0, 1, 5], ["q1", "q1", "q2"]),
                               (expected_q1 + 1) / 2)

    def test_gain_conventions_differ(self):
        # Put grade1 first, grade3 second. Explicit independent DCG expressions.
        discount = 1 / math.log2(3)
        exp_expected = (1 + 7 * discount) / (7 + discount)
        linear_expected = (1 + 3 * discount) / (3 + discount)
        self.assertAlmostEqual(mean_ndcg([1, 3], [2, 1], ["q", "q"]), exp_expected)
        self.assertAlmostEqual(mean_ndcg([1, 3], [2, 1], ["q", "q"], gain="linear"), linear_expected)

    def test_cutoff_and_zero_idcg(self):
        self.assertEqual(mean_ndcg([0, 3], [1, 0], ["q", "q"], k=1), 0)
        self.assertEqual(mean_ndcg([0, 0], [1, 0], ["q", "q"]), 0)
        self.assertEqual(mean_ndcg([3, 2, 1], [3, 2, 1], ["q"] * 3, k=2), 1)

    def test_ties_use_stable_rows_or_explicit_numeric_object_ids(self):
        self.assertEqual(mean_ndcg([0, 3], [1, 1], ["q", "q"], k=1), 0)
        self.assertEqual(mean_ndcg([0, 3], [1, 1], ["q", "q"], k=1,
                                   tie_breaker=["10", "2"]), 1)
        # Arbitrarily large IDs must not pass through float64 and lose bits.
        self.assertEqual(mean_ndcg([0, 3], [1, 1], ["q", "q"], k=1,
                                   tie_breaker=["9007199254740993", "9007199254740992"]), 1)
        self.assertEqual(mean_ndcg([0, 3], [1, 1], ["q", "q"], k=1,
                                   tie_breaker=["z", "a"]), 1)

    def test_permutation_invariance_with_explicit_tie_keys(self):
        labels = np.array([3, 0, 1, 2, 3])
        scores = np.array([.4, .4, .2, .9, .1])
        qids = np.array(["b", "a", "b", "a", "a"])
        objects = np.array(["7", "5", "3", "8", "1"])
        order = np.array([4, 3, 2, 0, 1])
        score1 = mean_ndcg(labels, scores, qids, tie_breaker=objects)
        score2 = mean_ndcg(labels[order], scores[order], qids[order], tie_breaker=objects[order])
        self.assertEqual(score1, score2)

    def test_metric_rejects_invalid_values(self):
        cases = [
            ([], [], [], {}), ([1], [], ["q"], {}),
            ([1], [np.nan], ["q"], {}), ([1], [np.inf], ["q"], {}),
            ([np.nan], [1], ["q"], {}), ([-1], [1], ["q"], {}),
            ([1], [1], ["q"], {"k": 0}), ([1], [1], ["q"], {"k": 1.5}),
            ([1], [1], ["q"], {"gain": "other"}),
            ([1], [1], ["q"], {"tie_breaker": []}),
            ([10000], [1], ["q"], {}),
        ]
        for labels, scores, qids, kwargs in cases:
            with self.subTest(labels=labels, scores=scores, kwargs=kwargs), self.assertRaises(ValueError):
                mean_ndcg(labels, scores, qids, **kwargs)


if __name__ == "__main__":
    unittest.main()
