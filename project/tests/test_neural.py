"""Small, model-free checks for neural input selection and resumable row safety."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from neural import (choose_body_window, make_passage, pair_fingerprint,
                    select_rows, topk_rows, load_checkpoint, save_checkpoint)


class NeuralTests(unittest.TestCase):
    def pairs(self):
        return SimpleNamespace(query_ids=np.array(["train:a", "train:a", "test:b", "test:b"]),
            doc_ids=np.array(["d1", "d2", "d3", "d4"]),
            object_ids=np.array(["", "", "1", "2"]),
            is_train=np.array([True, True, False, False]),
            query_text={"train:a": "кошка", "test:b": "Москва"})

    def test_query_window_reaches_late_relevant_text_and_keeps_original(self):
        body = " ".join(["нейтральный"] * 40 + ["Погода", "Москвы"] + ["нейтральный"] * 10)
        passage, start, end = choose_body_window("погода в Москве", body, 10, 5)
        self.assertIn("Погода Москвы", passage)
        self.assertGreater(start, 20)
        self.assertEqual(end - start, 10)

    def test_no_overlap_uses_head_and_title_is_not_duplicated(self):
        body = "один два три четыре пять шесть"
        self.assertEqual(choose_body_window("погода", body, 3, 2), ("один два три", 0, 3))
        self.assertEqual(make_passage("погода", "Заголовок", "", 3, 2), "Заголовок")

    def test_selection_intersects_rows_and_queries_without_labels(self):
        p = self.pairs()
        self.assertEqual(select_rows(p, "test", rows=np.array([False, True, True, False])).tolist(), [2])
        self.assertEqual(select_rows(p, query_ids=["train:a"]).tolist(), [0, 1])
        with self.assertRaises(ValueError):
            select_rows(p, rows=np.array([0, 0]))
        with self.assertRaises(ValueError):
            select_rows(p, rows=np.array([1.5]))

    def test_topk_ties_are_stable_with_explicit_keys(self):
        rows = topk_rows(["a", "b", "a", "b"], [1., 0., 1., 2.], 1, [8, 3, 2, 4])
        self.assertEqual(rows.tolist(), [2, 3])

    def test_pair_fingerprint_tracks_text_and_order(self):
        p = self.pairs(); a = pair_fingerprint(p)
        p.query_text["test:b"] = "Погода в Москве"
        self.assertNotEqual(pair_fingerprint(p), a)
        p = self.pairs(); p.doc_ids[[0, 1]] = p.doc_ids[[1, 0]]
        self.assertNotEqual(pair_fingerprint(p), a)

    def test_checkpoint_resume_never_turns_missing_scores_into_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.npz"
            scores = np.array([1.25, np.nan, -2.5], dtype=np.float32)
            mask = np.array([True, False, True])
            save_checkpoint(path, scores, mask, {"signature": "abc"})
            loaded, computed, _ = load_checkpoint(path, 3, "abc")
            self.assertTrue(np.isnan(loaded[1]))
            np.testing.assert_array_equal(computed, mask)
            np.testing.assert_allclose(loaded[computed], scores[mask])
            with self.assertRaises(ValueError):
                load_checkpoint(path, 3, "changed-model-or-data")
            with self.assertRaises(ValueError):
                load_checkpoint(path, 4, "abc")


if __name__ == "__main__":
    unittest.main()
