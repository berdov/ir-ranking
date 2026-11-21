import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import FeatureBundle, load_pairs
from features import build_features, proximity


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.cache = self.root / "cache"
        files = {
            "vkmarco-docs.tsv": "d1\thttps://example.ru/cat\tкоты\tкоты спят коты\n"
                                  "d2\thttps://example.ru/dog\tсобаки\tспят\n"
                                  "d3\thttps://example.ru/empty\tкоты\t\n",
            "vkmarco-doctrain-queries.tsv": "10\tкоты\n",
            "vkmarco-doctrain-qrels.tsv": "10 0 d1 3\n10 0 d2 0\n",
            "vkmarco-doceval-queries.tsv": "20\tкоты\n",
            "objects.csv": "ObjectId,QueryId,DocumentId\n7,20,d3\n2,20,d1\n",
            "sample_submission.csv": "ObjectId,Score\n2,0\n7,0\n",
        }
        for name, text in files.items():
            (self.data / name).write_text(text, encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def test_bm25f_uses_full_corpus_stats_and_shared_saturation(self):
        bundle = build_features(self.data, self.cache)
        column = bundle.feature_names.index("bm25f_default")
        idf = np.log1p((3 - 2 + .5) / (2 + .5))
        tf = 4 + 2 / (1 - .6 + .6 * 3 / (4 / 3))
        expected = idf * tf * 3 / (tf + 2)
        self.assertAlmostEqual(bundle.X[0, column], expected, places=6)
        self.assertEqual(bundle.metadata["documents"], 3)
        self.assertEqual(bundle.object_ids[~bundle.is_train].tolist(), ["7", "2"])
        with sqlite3.connect(self.cache / "documents.sqlite") as conn:
            self.assertEqual(conn.execute("select body from documents where doc_id='d1'").fetchone()[0], "коты спят коты")
        self.assertTrue(np.isfinite(bundle.X).all())
        self.assertEqual(len(bundle.feature_names), len(set(bundle.feature_names)))

    def test_cache_roundtrip_and_label_changes_do_not_change_features(self):
        first = build_features(self.data, self.cache)
        cached = build_features(self.data, self.cache)
        np.testing.assert_array_equal(first.X, cached.X)
        loaded = FeatureBundle.load(self.cache / "features.npz")
        np.testing.assert_array_equal(first.X, loaded.X)
        (self.data / "vkmarco-doctrain-qrels.tsv").write_text("10 0 d1 0\n10 0 d2 3\n")
        changed = build_features(self.data, self.cache)
        np.testing.assert_array_equal(first.X, changed.X)
        self.assertNotEqual(first.labels.tolist(), changed.labels.tolist())
        self.assertNotEqual(first.metadata["fingerprint"], changed.metadata["fingerprint"])

    def test_unknown_document_is_rejected(self):
        (self.data / "objects.csv").write_text("ObjectId,QueryId,DocumentId\n7,20,missing\n2,20,d1\n")
        with self.assertRaisesRegex(ValueError, "Unknown candidate"):
            build_features(self.data, self.cache)
        self.assertFalse((self.cache / "features.npz").exists())

    def test_profile_cannot_be_reused_as_complete(self):
        profile = build_features(self.data, self.cache, max_documents=1)
        self.assertFalse(profile.metadata["complete"])
        self.assertFalse((self.cache / "features.npz").exists())
        self.assertTrue((self.cache / "features.profile-1.npz").is_file())
        full = build_features(self.data, self.cache)
        self.assertTrue(full.metadata["complete"])
        self.assertEqual(len(full.X), 4)

    def test_duplicate_qrels_and_train_test_overlap_rejected(self):
        path = self.data / "vkmarco-doctrain-qrels.tsv"
        path.write_text(path.read_text() + "10 0 d1 0\n")
        with self.assertRaisesRegex(ValueError, "Duplicate training"):
            load_pairs(self.data)
        (self.data / "vkmarco-doceval-queries.tsv").write_text("10\tсобаки\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_pairs(self.data)

    def test_proximity_covers_all_terms_inside_field(self):
        density, first, last = proximity({"a": [0, 8], "b": [7]}, ["a", "b"], 10)
        self.assertEqual(density, 1.)
        self.assertEqual(first, 0.)
        self.assertEqual(last, .8)
        self.assertEqual(proximity({"a": [0]}, ["a", "b"], 10)[0], 0.)


if __name__ == "__main__":
    unittest.main()
