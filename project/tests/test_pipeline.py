import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import dataset_hash, ensure_split, rows_for
from pipeline import aligned_predictions, final_audit, neural_tune
from ranking import NDCG, subset, write_json
from test_ranking import toy_bundle


class AlignmentTests(unittest.TestCase):
    def test_join_reorders_by_both_keys_and_rejects_bad_labels(self):
        bundle = toy_bundle()
        rows = np.arange(12)
        order = rows[::-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            values = dict(query_ids=bundle.query_ids[order], doc_ids=bundle.doc_ids[order],
                          labels=bundle.labels[order], selected=order * .1, baseline=order * .2)
            np.savez(path, **values)
            result = aligned_predictions(path, bundle, rows)
            np.testing.assert_allclose(result["selected"], rows * .1)
            values["labels"] = (values["labels"] + 1) % 4
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "labels differ"):
                aligned_predictions(path, bundle, rows)


class FrozenPipelineTests(unittest.TestCase):
    def run_fixture(self, directory, neural_helpful, corrupt_before_audit=False):
        root = Path(directory)
        out = root / "reports"
        config_dir = root / "configs"
        cache = root / "cache"
        out.mkdir(); cache.mkdir()
        bundle = toy_bundle(queries=40)
        bundle.metadata = {"complete": True, "fingerprint": {"synthetic": True}}
        split = ensure_split(bundle, out)
        fit_rows = rows_for(bundle, split["query_ids"]["fit"])
        tune_rows = rows_for(bundle, split["query_ids"]["tune"])
        holdout_rows = rows_for(bundle, split["query_ids"]["holdout"])
        tune = subset(bundle, tune_rows)
        selection = {"format_version": 1, "name": "fixture", "seed": 42,
                     "input_feature_names": bundle.feature_names, "split_sha256": split["sha256"],
                     "fit_sha256": dataset_hash(subset(bundle, fit_rows)),
                     "tune_sha256": dataset_hash(tune),
                     "tune_ndcg": NDCG(tune.labels, tune.query_ids, tune.doc_ids).mean(tune.X[:, 0])}
        write_json(out / "selection.json", selection)
        np.savez(out / "tune-predictions.npz", query_ids=tune.query_ids, doc_ids=tune.doc_ids,
                 labels=tune.labels, selected=tune.X[:, 0], baseline=tune.X[:, 0])
        requests = []

        def fake_score(pairs, cache_dir, model_dir, rows, **kwargs):
            rows = np.asarray(rows)
            requests.append(rows.copy())
            scores = np.full(len(bundle.labels), np.nan, dtype=np.float32)
            available = np.zeros(len(bundle.labels), dtype=bool)
            # Synthetic oracle only for testing fusion/masking; never used in production.
            scores[rows] = bundle.labels[rows] if neural_helpful else bundle.X[rows, 0]
            available[rows] = True
            return scores, available, {"runs": [{"rows": len(rows), "seconds": 1.}]}

        def fake_audit(bundle, selection, out_dir, threads):
            holdout = subset(bundle, holdout_rows)
            scores = holdout.X[:, 0]
            np.savez(Path(out_dir) / "holdout-predictions.npz", query_ids=holdout.query_ids,
                     doc_ids=holdout.doc_ids, labels=holdout.labels,
                     selected=scores, baseline=scores)
            return {"selected_ndcg": NDCG(holdout.labels, holdout.query_ids, holdout.doc_ids).mean(scores)}

        with patch("pipeline._load_inputs", return_value=(bundle, object(), selection, split)), \
                patch("pipeline._score_rows", side_effect=fake_score), \
                patch("pipeline.audit_selection", side_effect=fake_audit):
            tuned = neural_tune(root, cache, out, root / "nonexistent-model", config_dir=config_dir)
            np.testing.assert_array_equal(requests[0], tune_rows)
            self.assertFalse((out / "final-audit-lock.json").exists())
            if corrupt_before_audit:
                path = config_dir / "fusion_config.json"
                altered = json.loads(path.read_text())
                altered["config"]["neural_weight"] = .123
                write_json(path, altered)
                with self.assertRaisesRegex(ValueError, "modified"):
                    final_audit(root, cache, out, root / "nonexistent-model", config_dir=config_dir)
                self.assertEqual(len(requests), 1)
                return
            audited = final_audit(root, cache, out, root / "nonexistent-model", config_dir=config_dir)
            self.assertEqual(audited["used_for_selection"], False)
            self.assertTrue((out / "fusion-freeze.json").is_file())
            self.assertTrue((out / "final-audit-lock.json").is_file())
            with self.assertRaisesRegex(ValueError, "no longer allowed"):
                neural_tune(root, cache, out, root / "nonexistent-model", config_dir=config_dir)
            self.assertEqual(audited, final_audit(root, cache, out, root / "nonexistent-model", config_dir=config_dir))
            if neural_helpful:
                self.assertEqual(len(requests), 2)
                self.assertTrue(set(requests[1]) <= set(holdout_rows))
                self.assertGreater(audited["selected_ndcg"], audited["lexical_ndcg"])
                self.assertEqual(len(requests[1]), audited["neural_pairs"])
            else:
                self.assertEqual(len(requests), 1)
                self.assertEqual(tuned["config"]["neural_weight"], 0)
                self.assertEqual(audited["neural_pairs"], 0)
                self.assertIsNone(audited["neural_metadata"])
                self.assertEqual(audited["selected_ndcg"], audited["lexical_ndcg"])

    def test_frozen_pipeline_scores_only_tune_then_needed_holdout_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_fixture(directory, True)

    def test_zero_neural_weight_skips_audit_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_fixture(directory, False)

    def test_changed_fusion_is_rejected_before_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_fixture(directory, True, corrupt_before_audit=True)


if __name__ == "__main__":
    unittest.main()
