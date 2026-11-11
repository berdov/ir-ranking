"""Selection-contract tests; no fitting or official competition data needed."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import Dataset
import experiments


def dataset(labels, query_ids, X=None, object_ids=None):
    labels = np.asarray(labels, dtype=np.int32)
    if X is None:
        X = np.zeros((len(labels), 1), dtype=np.float32)
    return Dataset(np.asarray(X, dtype=np.float32), np.asarray(query_ids), labels,
                   object_ids, ("f1",))


def component(family, *, seed=None):
    result = {"family": family, "features": "raw", "iterations": 200,
              "gain": "exp2", "params": {}}
    if seed is not None:
        result["seed"] = seed
    return result


class ExperimentTests(unittest.TestCase):
    def test_three_way_split_is_complete_disjoint_and_row_order_independent(self):
        queries = np.repeat([f"q{i:04d}" for i in range(7000)], 2)
        split = experiments.make_split(queries, seed=42)
        self.assertEqual({name: len(ids) for name, ids in split.items()},
                         {"fit": 5000, "tuning": 1000, "holdout": 1000})
        sets = {key: set(value) for key, value in split.items()}
        self.assertFalse(sets["fit"] & sets["tuning"])
        self.assertFalse(sets["fit"] & sets["holdout"])
        self.assertFalse(sets["tuning"] & sets["holdout"])
        self.assertEqual(set.union(*sets.values()), set(queries))
        permuted = np.random.default_rng(3).permutation(queries)
        repeated = experiments.make_split(permuted, seed=42)
        for name in split:
            np.testing.assert_array_equal(split[name], repeated[name])
        with self.assertRaises(ValueError):
            experiments.make_split(np.arange(6))

    def test_official_baseline_is_raw_200_default_params_seed_zero(self):
        for gain in ("exp2", "linear"):
            baseline = experiments.candidates(gain)[0]
            self.assertEqual(baseline, {
                "name": "baseline_yetirank200", "family": "catboost", "features": "raw",
                "iterations": 200, "gain": gain, "params": {}, "seed": 0,
            })
        # Smoke configurations must not mutate the real search configuration.
        self.assertEqual(experiments.candidates(quick=True)[0]["iterations"], 8)
        self.assertEqual(experiments.candidates()[0]["iterations"], 200)

    def test_selects_correct_ensemble_and_preserves_component_seeds(self):
        tune = dataset([3, 1, 0, 3, 1, 0], ["q1"] * 3 + ["q2"] * 3)
        cases = {"weak": component("catboost", seed=0),
                 "a": component("catboost", seed=17),
                 "b": component("lightgbm", seed=23)}
        original_cases = copy.deepcopy(cases)
        predictions = {
            "weak": np.array([0, 1, 2, 0, 1, 2]),
            "a": np.array([3, 2, 1, 1, 2, 0]),
            "b": np.array([1, 2, 0, 3, 2, 1]),
        }
        (value, name, config), choices = experiments.choose(cases, predictions, tune, seed=99)
        self.assertEqual(value, 1.)
        self.assertEqual(name, "a+b:zscore:0.5")
        self.assertEqual(config["seed"], 99)
        self.assertEqual(config["score_normalization"], "zscore")
        self.assertEqual([item["seed"] for item in config["components"]], [17, 23])
        self.assertEqual([item["weight"] for item in config["components"]], [.5, .5])
        self.assertEqual(sum(item["weight"] for item in config["components"]), 1.)
        self.assertEqual(cases, original_cases)
        self.assertEqual(len(choices), 3 + 2 * 3)

    def test_exact_tie_prefers_single_candidate_and_choice_ignores_features_and_ids(self):
        labels = [3, 1, 0]
        tune = dataset(labels, ["q"] * 3, object_ids=np.array(["100", "200", "300"]))
        changed_irrelevant_fields = dataset(labels, ["q"] * 3,
                                             X=[[np.nan], [999999], [-999999]],
                                             object_ids=np.array(["3", "2", "1"]))
        cases = {"first": component("catboost", seed=0), "second": component("lightgbm")}
        predictions = {"first": np.array([3, 2, 1]), "second": np.array([30, 20, 10])}
        result = experiments.choose(cases, predictions, tune)
        changed = experiments.choose(cases, predictions, changed_irrelevant_fields)
        self.assertEqual(result, changed)
        (value, name, config), _ = result
        self.assertEqual((value, name), (1., "first"))
        self.assertEqual(len(config["components"]), 1)
        self.assertEqual(config["components"][0]["seed"], 0)

    def test_main_selection_invariant_to_holdout_labels_and_unreadable_test(self):
        # Mock model fitting only: retain the real split, selection, metrics and
        # on-disk decision. Changing all holdout labels must not change it.
        qids = np.repeat([f"q{i:02d}" for i in range(28)], 3)
        labels = np.tile([3, 1, 0], 28)
        X = np.tile([[3.], [2.], [1.]], (28, 1)).astype(np.float32)
        base = dataset(labels, qids, X)
        split = experiments.make_split(qids, 42)
        holdout_mask = np.isin(qids, split["holdout"])
        altered_labels = labels.copy()
        altered_labels[holdout_mask] = 0
        altered = dataset(altered_labels, qids, X)
        train_sets = []
        selection_snapshots = []
        for input_data in (base, altered):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data_dir, output_dir = root / "data", root / "out"
                data_dir.mkdir()
                # If the experiment starts reading test.csv, this is not a CSV.
                (data_dir / "test.csv").write_bytes(b"\xffnot valid UTF-8 CSV")
                calls = []

                def mock_load(path):
                    self.assertEqual(path, data_dir / "train.csv")
                    return input_data

                def mock_fit(train, config, validation=None, seed=42, threads=4):
                    calls.append((train.query_ids.copy(), None if validation is None else
                                  validation.query_ids.copy(), copy.deepcopy(config), seed))
                    # The seed passed to baseline fitting must agree with its
                    # per-component seed instead of the global selection seed.
                    self.assertEqual(seed, config.get("seed", 42))
                    self.assertFalse(set(train.query_ids) & set(split["holdout"]))
                    if validation is not None:
                        self.assertFalse(set(train.query_ids) & set(validation.query_ids))
                        self.assertEqual(set(validation.query_ids), set(split["tuning"]))
                    return object(), dict(config)

                def mock_predict(model, data, config, threads=4):
                    return data.X[:, 0].astype(np.float64)

                original_write = experiments.write_json

                def observed_write(path, value):
                    if path.name == "best_config.json":
                        # Candidate fitting is complete, but holdout auditing
                        # has not started at the instant the decision is saved.
                        self.assertEqual(len(calls), len(experiments.candidates()))
                    original_write(path, value)

                with patch.object(experiments, "load_train", side_effect=mock_load), \
                     patch.object(experiments, "fit_component", side_effect=mock_fit), \
                     patch.object(experiments, "predict_component", side_effect=mock_predict), \
                     patch.object(experiments, "save_models"), \
                     patch.object(experiments, "digest", return_value="fixture"), \
                     patch.object(experiments, "write_json", side_effect=observed_write), \
                     patch.object(sys, "argv", ["experiments.py", str(data_dir), "--output_dir", str(output_dir)]), \
                     redirect_stdout(io.StringIO()):
                    experiments.main()
                first = calls[0]
                self.assertIsNone(first[1])  # Baseline gets no validation pool.
                self.assertEqual(first[2]["iterations"], 200)
                self.assertEqual(first[2]["params"], {})
                self.assertEqual(first[3], 0)
                selection_snapshots.append(json.loads((output_dir / "best_config.json").read_text()))
                train_sets.append([set(item[0]) for item in calls])
                report = json.loads((output_dir / "experiments.json").read_text())
                self.assertFalse(report["holdout"]["used_for_selection"])
        self.assertEqual(selection_snapshots[0], selection_snapshots[1])
        self.assertEqual(train_sets[0], train_sets[1])


if __name__ == "__main__":
    unittest.main()
