import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import dataset_hash, rows_for, split_manifest
from features import build_features
from portability import adapt_selection_for_runtime, build_manifest
from ranking import subset


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        files = {
            "vkmarco-docs.tsv": "d1\thttps://example.ru/cat\tкоты\tкоты спят\n"
                                  "d2\thttps://example.ru/dog\tсобаки\tсобаки спят\n",
            "vkmarco-doctrain-queries.tsv": "".join(f"{i}\tкоты {i}\n" for i in range(1, 8)),
            "vkmarco-doctrain-qrels.tsv": "".join(f"{i} 0 d1 3\n{i} 0 d2 0\n" for i in range(1, 8)),
            "vkmarco-doceval-queries.tsv": "20\tкоты\n",
            "objects.csv": "ObjectId,QueryId,DocumentId\n7,20,d1\n2,20,d2\n",
            "sample_submission.csv": "ObjectId,Score\n2,0\n7,0\n",
        }
        for name, text in files.items():
            (self.data / name).write_text(text, encoding="utf-8")
        self.bundle = build_features(self.data, self.root / "cache")
        self.split = split_manifest(self.bundle, 42)
        self.selection = {
            "format_version": 1, "seed": 42, "split_sha256": self.split["sha256"],
            "input_feature_names": self.bundle.feature_names,
            "components": [{"family": "catboost", "iterations": 641, "params": {}}],
        }
        for name in ("fit", "tune"):
            rows = rows_for(self.bundle, self.split["query_ids"][name])
            self.selection[name + "_sha256"] = dataset_hash(subset(self.bundle, rows))
        self.manifest = build_manifest(self.bundle, self.selection, self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def adapt(self, bundle=None, selection=None, manifest=None):
        return adapt_selection_for_runtime(bundle or self.bundle, selection or self.selection,
                                           self.data, manifest or self.manifest)

    def test_exact_inputs_preserve_selection_and_hash(self):
        original = copy.deepcopy(self.selection)
        runtime, report = self.adapt()
        self.assertEqual(runtime, original)
        self.assertEqual(self.selection, original)
        self.assertFalse(report["adapted"])
        self.assertEqual(report["original_selection_sha256"], report["runtime_selection_sha256"])

    def test_one_ulp_source_bound_adaptation_changes_only_hashes(self):
        bundle = copy.deepcopy(self.bundle)
        row = rows_for(bundle, self.split["query_ids"]["fit"])[0]
        column = bundle.feature_names.index("lm_body_mu1500")
        bundle.X[row, column] = np.nextafter(bundle.X[row, column], np.float32(np.inf))
        original = copy.deepcopy(self.selection)
        runtime, report = self.adapt(bundle=bundle)
        changed = [key for key in runtime if runtime[key] != original[key]]
        self.assertEqual(changed, ["fit_sha256"])
        self.assertEqual(self.selection, original)
        self.assertTrue(report["adapted"])
        self.assertFalse(report["tuning_performed"])
        self.assertFalse(report["numerical_tolerance_checked"])
        with self.assertRaisesRegex(ValueError, "original selection"):
            build_manifest(bundle, self.selection, self.data)

    def test_changed_small_input_is_rejected(self):
        path = self.data / "vkmarco-doctrain-queries.tsv"
        path.write_text(path.read_text() + "8\tчужой запрос\n")
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.adapt()

    def test_changed_corpus_metadata_digest_is_rejected(self):
        bundle = copy.deepcopy(self.bundle)
        bundle.metadata["corpus_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "input_files"):
            self.adapt(bundle=bundle)

    def test_changed_source_is_rejected_even_with_matching_local_cache(self):
        bundle = copy.deepcopy(self.bundle)
        bundle.metadata["fingerprint"]["source_sha256"]["features.py"] = "f" * 64
        with patch("portability.source_fingerprint", return_value=bundle.metadata["fingerprint"]):
            with self.assertRaisesRegex(ValueError, "feature_implementation"):
                self.adapt(bundle=bundle)

    def test_changed_label_document_object_and_query_ids_are_rejected(self):
        for key in ("labels", "doc_ids", "object_ids", "query_ids"):
            with self.subTest(key=key):
                bundle = copy.deepcopy(self.bundle)
                array = getattr(bundle, key)
                index = len(array) - 1 if key == "object_ids" else 0
                array[index] = 1 if key == "labels" else "9"
                with self.assertRaises(ValueError):
                    self.adapt(bundle=bundle)

    def test_changed_schema_and_hyperparameters_are_rejected(self):
        bundle = copy.deepcopy(self.bundle)
        bundle.feature_names[0] = "changed"
        with self.assertRaisesRegex(ValueError, "schema"):
            self.adapt(bundle=bundle)
        selection = copy.deepcopy(self.selection)
        selection["components"][0]["iterations"] += 1
        with self.assertRaisesRegex(ValueError, "selection differs"):
            self.adapt(selection=selection)

    def test_changed_numpy_version_or_manifest_is_rejected(self):
        with patch("portability.np.__version__", "999"):
            with self.assertRaisesRegex(ValueError, "versions"):
                self.adapt()
        manifest = copy.deepcopy(self.manifest)
        manifest["policy"] = "different"
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.adapt(manifest=manifest)


if __name__ == "__main__":
    unittest.main()
