"""Bind frozen model choices to input provenance across floating-point platforms.

This is a source-provenance check, not an elementwise tolerance check on X.
The full corpus digest was computed during feature extraction; cache reuse
revalidates its size, timestamp and edge digest, as build_features does. Other
input files and implementation files are hashed completely on every call.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from data import FeatureBundle, file_hash
from experiments import dataset_hash, rows_for, split_manifest
from features import source_fingerprint
from ranking import json_hash, subset, write_json

FORMAT_VERSION = 1
ROOT = Path(__file__).resolve().parent
CORE_FILES = ("ranking.py", "experiments.py", "fusion.py", "neural.py", "pipeline.py")
IDENTITY_ARRAYS = ("labels", "query_ids", "doc_ids", "object_ids", "is_train")


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _provenance(bundle, selection, data_dir):
    bundle.validate()
    if not bundle.metadata.get("complete"):
        raise ValueError("Portability requires a complete feature bundle")
    current = source_fingerprint(data_dir)
    recorded = bundle.metadata.get("fingerprint")
    if recorded != current:
        raise ValueError("Feature cache provenance differs from current input files or implementation")
    corpus_sha = bundle.metadata.get("corpus_sha256")
    if (not isinstance(corpus_sha, str) or len(corpus_sha) != 64
            or any(c not in "0123456789abcdef" for c in corpus_sha)):
        raise ValueError("Missing full corpus SHA-256 from feature extraction")
    sources = {}
    for name, info in current["files"].items():
        sources[name] = {"size": info["size"], "sha256":
                         corpus_sha if name == "vkmarco-docs.tsv" else info["sha256"]}
    split = split_manifest(bundle, selection["seed"])
    if split["sha256"] != selection["split_sha256"]:
        raise ValueError("Query split differs from the frozen selection")
    if list(bundle.feature_names) != selection["input_feature_names"]:
        raise ValueError("Feature schema differs from the frozen selection")
    return {
        "input_files": sources,
        "feature_schema_version": current["schema_version"],
        "feature_implementation_sha256": current["source_sha256"],
        "core_implementation_sha256": {name: file_hash(ROOT / name) for name in CORE_FILES},
        "versions": {"numpy": np.__version__, "PyStemmer": current["stemmer_version"]},
        "feature_names": list(bundle.feature_names),
        "matrix_shape": list(bundle.X.shape), "matrix_dtype": str(bundle.X.dtype),
        "row_identity_sha256": {key: _array_hash(getattr(bundle, key)) for key in IDENTITY_ARRAYS},
        "split_sha256": split["sha256"],
        "corpus_statistics": {key: bundle.metadata.get(key) for key in
            ("documents", "candidate_documents", "query_vocabulary_size",
             "field_token_totals", "field_average_lengths")},
    }, split


def _partition_hashes(bundle, split):
    return {name + "_sha256": dataset_hash(subset(bundle, rows_for(bundle, split["query_ids"][name])))
            for name in ("fit", "tune")}


def build_manifest(bundle, selection, data_dir):
    """Create a published manifest only from the exact selected feature bytes."""
    provenance, split = _provenance(bundle, selection, data_dir)
    hashes = _partition_hashes(bundle, split)
    for key, actual in hashes.items():
        if actual != selection.get(key):
            raise ValueError(f"Cannot publish manifest: original selection {key} differs")
    payload = {"format_version": FORMAT_VERSION, "selection_sha256": json_hash(selection),
               "original_partition_hashes": hashes, "provenance": provenance,
               "policy": "same verified inputs and implementation; platform float bytes may differ"}
    return dict(payload, manifest_sha256=json_hash(payload))


def adapt_selection_for_runtime(bundle, selection, data_dir, manifest):
    """Return (selection_copy, report); change only fit/tune byte fingerprints.

    Trust boundary: bundle must come from the ordinary validated feature cache
    or a fresh extraction. This does not certify arbitrary edited X values or
    impose a numerical error bound. No parameters or model choices are selected.
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    manifest = copy.deepcopy(manifest)
    expected_sha = manifest.pop("manifest_sha256", None)
    if expected_sha != json_hash(manifest) or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Invalid portability manifest")
    original_sha = json_hash(selection)
    if original_sha != manifest["selection_sha256"]:
        raise ValueError("Frozen selection differs from the published portability manifest")
    provenance, split = _provenance(bundle, selection, data_dir)
    for key, expected in manifest["provenance"].items():
        if provenance.get(key) != expected:
            raise ValueError(f"Portability provenance mismatch: {key}")
    if set(provenance) != set(manifest["provenance"]):
        raise ValueError("Portability provenance schema differs")
    for key, expected in manifest["original_partition_hashes"].items():
        if selection.get(key) != expected:
            raise ValueError(f"Original selection partition differs: {key}")
    actual = _partition_hashes(bundle, split)
    runtime = copy.deepcopy(selection)
    runtime.update(actual)
    changed = [key for key in actual if actual[key] != selection[key]]
    report = {
        "manifest_sha256": expected_sha,
        "original_selection_sha256": original_sha,
        "runtime_selection_sha256": json_hash(runtime),
        "adapted": bool(changed), "changed_fields": changed,
        "original_partition_hashes": {key: selection[key] for key in actual},
        "runtime_partition_hashes": actual,
        "provenance_verified": True, "tuning_performed": False,
        "validation_policy": manifest["policy"],
        "numerical_tolerance_checked": False,
        "corpus_validation": "full SHA-256 recorded during extraction; current size/mtime/edges revalidated",
    }
    return runtime, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selection_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=ROOT.parent.parent / "data" / "project")
    args = parser.parse_args()
    bundle = FeatureBundle.load(args.features)
    selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
    manifest = build_manifest(bundle, selection, args.data_dir)
    write_json(args.output_file, manifest)
    print(json.dumps({"manifest": str(args.output_file), "sha256": manifest["manifest_sha256"]}))


if __name__ == "__main__":
    main()
