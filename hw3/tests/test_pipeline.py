"""Offline feature and native-model checks on small generated query groups."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import Dataset, load_test, load_train
from features import normalize_scores, transform
from modeling import (ensemble_predict, fit_component, load_models,
                      predict_component, save_models)


def tiny_component(family):
    params = ({"num_leaves": 4, "min_data_in_leaf": 1, "min_data_in_bin": 1,
               "feature_fraction": 1.0, "learning_rate": .1}
              if family == "lightgbm" else {"depth": 2, "learning_rate": .1})
    return {"family": family, "features": "relative", "gain": "exp2",
            "iterations": 2, "weight": 1.0, "params": params}


def training_fixture():
    # The row order deliberately interleaves queries. There are no text fields
    # or source IDs; the fixture is entirely synthetic and independent of data.
    labels = np.tile([0, 3, 1, 2], 8).astype(np.int32)
    query_ids = np.repeat([f"train-{i}" for i in range(8)], 4)
    first = labels * 2.0 + np.repeat(np.arange(8) * 10.0, 4)
    second = np.tile([0.0, 1.0, np.nan, 1.0], 8)
    third = np.repeat(np.arange(8) + 3.0, 4)
    X = np.column_stack([first, second, third]).astype(np.float32)
    permutation = np.random.default_rng(123).permutation(len(labels))
    return Dataset(X[permutation], query_ids[permutation], labels[permutation],
                   None, ("signal", "optional", "constant"))


def prediction_fixture():
    return Dataset(
        np.array([[21, 1, 4], [3, np.nan, 2], [25, 0, 4],
                  [5, 1, 2], [23, 1, 4], [1, 0, 2]], dtype=np.float32),
        np.array(["beta", "alpha", "beta", "alpha", "beta", "alpha"]),
        None, np.array(["0009", "9007199254740993", "7", "NA", "03", "last"]),
        ("signal", "optional", "constant"),
    )


def write_dataset(path, dataset):
    train = dataset.labels is not None
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((["QueryId", "Label"] if train else ["ObjectId", "QueryId"])
                        + list(dataset.feature_names))
        for i, row in enumerate(dataset.X):
            keys = ([dataset.query_ids[i], int(dataset.labels[i])] if train
                    else [dataset.object_ids[i], dataset.query_ids[i]])
            writer.writerow(keys + ["" if np.isnan(x) else float(x) for x in row])


class FeatureTests(unittest.TestCase):
    def test_relative_values_ties_constants_missing_and_singletons(self):
        query_ids = np.array(["a", "b", "a", "a", "b", "singleton"])
        X = np.array([[1, 7, np.nan], [100, 4, 1], [1, 7, np.nan],
                      [3, np.nan, np.nan], [200, 4, 3], [8, 2, 5]], dtype=np.float32)
        result = transform(X, query_ids, "relative")
        np.testing.assert_equal(result[:, :3], X)
        np.testing.assert_allclose(result[:, 3:6],
                                   [[.25, 0, np.nan], [0, 0, 0], [.25, 0, np.nan],
                                    [1, np.nan, np.nan], [1, 0, 1], [0, 0, 0]],
                                   equal_nan=True)
        np.testing.assert_allclose(result[:, 6:],
                                   [[0, 0, np.nan], [0, 0, 0], [0, 0, np.nan],
                                    [1, np.nan, np.nan], [1, 0, 1], [0, 0, 0]],
                                   equal_nan=True)
        self.assertEqual(result.dtype, np.float32)

    def test_other_queries_do_not_change_features(self):
        dataset = training_fixture()
        selected = dataset.query_ids == "train-2"
        expected = transform(dataset.X[selected], dataset.query_ids[selected], "relative")
        all_rows = transform(dataset.X, dataset.query_ids, "relative")
        np.testing.assert_equal(all_rows[selected], expected)
        poisoned = dataset.X.copy()
        poisoned[~selected] = 1e20
        np.testing.assert_equal(transform(poisoned, dataset.query_ids, "relative")[selected],
                                expected)
        # Changing the spelling of group identifiers has no effect when group
        # membership stays the same, and does not add any feature columns.
        renamed = np.array(["prefix/" + q for q in dataset.query_ids])
        np.testing.assert_equal(transform(dataset.X, renamed, "relative"), all_rows)
        self.assertEqual(all_rows.shape[1], dataset.X.shape[1] * 3)

    def test_transform_is_row_permutation_equivariant(self):
        dataset = training_fixture()
        order = np.arange(len(dataset.query_ids))[::-1]
        for mode in ("raw", "rank", "relative"):
            with self.subTest(mode=mode):
                original = transform(dataset.X, dataset.query_ids, mode)
                shuffled = transform(dataset.X[order], dataset.query_ids[order], mode)
                np.testing.assert_equal(shuffled, original[order])

    def test_score_normalization_is_query_local_and_finite(self):
        scores = np.array([10, 2, 10, 4, 99], dtype=float)
        queries = np.array(["constant", "range", "constant", "range", "single"])
        np.testing.assert_allclose(normalize_scores(scores, queries, "zscore"), [0, -1, 0, 1, 0])
        np.testing.assert_allclose(normalize_scores(scores, queries, "rank"), [.5, 0, .5, 1, 0])
        np.testing.assert_equal(normalize_scores(scores, queries, "none"), scores)
        for invalid in ([1, np.nan], [1, np.inf]):
            with self.assertRaises(ValueError):
                normalize_scores(invalid, ["q", "q"])

    def test_invalid_transform_inputs_are_rejected(self):
        cases = [([[1], [2]], ["q"], "raw"), ([[np.inf]], ["q"], "rank"),
                 ([1], ["q"], "raw"), ([[1]], ["q"], "unknown")]
        for X, queries, mode in cases:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                transform(X, queries, mode)


class NativeModelTests(unittest.TestCase):
    def test_both_families_fit_reproducibly_and_round_trip_without_pickle(self):
        train, test = training_fixture(), prediction_fixture()
        with tempfile.TemporaryDirectory() as directory:
            for family in ("lightgbm", "catboost"):
                with self.subTest(family=family):
                    component = tiny_component(family)
                    model, fitted = fit_component(train, component, seed=42, threads=1)
                    refit, _ = fit_component(train, component, seed=42, threads=1)
                    expected = predict_component(model, test, fitted, threads=1)
                    np.testing.assert_array_equal(
                        expected, predict_component(refit, test, fitted, threads=1))
                    self.assertTrue(np.isfinite(expected).all())
                    # A trained model must use values/group membership only,
                    # never the spelling or numeric magnitude of object IDs.
                    renamed = Dataset(test.X, np.char.add("renamed-", test.query_ids), None,
                                      np.array(["changed"] * len(test.query_ids)), test.feature_names)
                    np.testing.assert_array_equal(
                        expected, predict_component(model, renamed, fitted, threads=1))
                    config = {"seed": 42, "score_normalization": "none", "components": [fitted]}
                    path = Path(directory) / f"{family}.bin"
                    save_models(path, [model], config, train.feature_names, {"fixture": True})
                    with zipfile.ZipFile(path) as archive:
                        suffix = "cbm" if family == "catboost" else "txt"
                        self.assertEqual(set(archive.namelist()), {"metadata.json", f"model_0.{suffix}"})
                    loaded, metadata = load_models(path)
                    self.assertEqual(tuple(metadata["feature_names"]), train.feature_names)
                    np.testing.assert_array_equal(
                        expected, ensemble_predict(loaded, test, metadata["config"], threads=1))


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.train_dir = self.directory / "train_only"
        self.predict_dir = self.directory / "test_only"
        self.train_dir.mkdir()
        self.predict_dir.mkdir()
        self.train, self.test = training_fixture(), prediction_fixture()
        write_dataset(self.train_dir / "train.csv", self.train)
        write_dataset(self.predict_dir / "test.csv", self.test)
        self.config_path = self.directory / "tiny_config.json"
        self.config_path.write_text(json.dumps({
            "seed": 42, "score_normalization": "zscore",
            "components": [tiny_component("lightgbm"), tiny_component("catboost")],
        }), encoding="utf-8")
        self.model_path = self.directory / "model.bin"

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *arguments, success=True):
        completed = subprocess.run([sys.executable, str(ROOT / "solution.py"), *map(str, arguments)],
                                   cwd=self.directory, capture_output=True, text=True, timeout=90)
        if success:
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        else:
            self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed

    def fit(self):
        return self.run_cli("--train", "--config_file", self.config_path, "--model_file",
                            self.model_path, "--threads", "1", self.train_dir)

    def test_two_cli_modes_use_only_their_own_csv_and_preserve_object_ids(self):
        self.assertFalse((self.train_dir / "test.csv").exists())
        self.assertFalse((self.predict_dir / "train.csv").exists())
        trained = self.fit()
        self.assertEqual(json.loads(trained.stdout.splitlines()[-1])["mode"], "train")
        first = self.directory / "first.csv"
        second = self.directory / "second.csv"
        for destination in (first, second):
            # The config path is deliberately absent: inference must read the
            # trained archive, never a new or modified external configuration.
            self.run_cli("--model_file", self.model_path, "--submission_file", destination,
                         "--config_file", self.directory / "not-present.json", "--threads", "1",
                         self.predict_dir)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with first.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["ObjectId", "Score"])
            rows = list(reader)
        self.assertEqual([row["ObjectId"] for row in rows], self.test.object_ids.tolist())
        scores = np.array([float(row["Score"]) for row in rows])
        self.assertTrue(np.isfinite(scores).all())
        models, metadata = load_models(self.model_path)
        loaded_test = load_test(self.predict_dir / "test.csv", self.test.feature_names)
        np.testing.assert_array_equal(scores, ensemble_predict(models, loaded_test,
                                                               metadata["config"], threads=1))
        self.assertEqual(metadata["feature_names"], list(self.train.feature_names))
        self.assertEqual(metadata["training"]["queries"], 8)
        self.assertEqual(metadata["training"]["rows"], 32)
        self.assertEqual(load_train(self.train_dir / "train.csv").feature_names,
                         ("signal", "optional", "constant"))

    def test_refuses_to_overwrite_inputs_config_and_loaded_model(self):
        for target in (self.train_dir / "train.csv", self.config_path):
            before = target.read_bytes()
            result = self.run_cli("--train", "--config_file", self.config_path, "--model_file",
                                  target, "--threads", "1", self.train_dir, success=False)
            self.assertIn("must not overwrite", result.stderr)
            self.assertEqual(target.read_bytes(), before)
        # Overwrite protection is checked before model deserialization.
        self.model_path.write_bytes(b"sentinel model bytes")
        for target in (self.predict_dir / "test.csv", self.model_path):
            before = target.read_bytes()
            result = self.run_cli("--model_file", self.model_path, "--submission_file", target,
                                  "--threads", "1", self.predict_dir, success=False)
            self.assertIn("must not overwrite", result.stderr)
            self.assertEqual(target.read_bytes(), before)

    def test_bad_prediction_input_does_not_replace_existing_submission(self):
        self.fit()
        output = self.directory / "existing.csv"
        output.write_bytes(b"keep the previous output\n")
        path = self.predict_dir / "test.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("signal", "unexpected_feature", 1),
                        encoding="utf-8")
        failed = self.run_cli("--model_file", self.model_path, "--submission_file", output,
                              "--threads", "1", self.predict_dir, success=False)
        self.assertIn("feature names/order differ", failed.stderr)
        self.assertEqual(output.read_bytes(), b"keep the previous output\n")


if __name__ == "__main__":
    unittest.main()
