#!/usr/bin/env python3
"""Select neural fusion on tune queries, then audit the frozen final pipeline.

``neural-tune`` never evaluates holdout labels. ``audit`` first verifies the
saved configuration hashes, freezes the experiment, and only then evaluates
the selected lexical/neural combination on held-out queries. Inference uses
only local, pinned model assets and a resumable device/batch-specific cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from data import FeatureBundle, file_hash, load_pairs
from experiments import (audit_selection, dataset_hash, ensure_split, paired_bootstrap,
                         rows_for)
from fusion import fuse_scores, required_rows, select_fusion
from neural import (MODEL_ID, MODEL_REVISION, VERSION as NEURAL_VERSION,
                    WEIGHTS_SHA256, score_pairs)
from ranking import METRIC, NDCG, json_hash, subset, write_json


ROOT = Path(__file__).resolve().parent
FORMAT_VERSION = 1


def neural_configuration():
    """Mathematical inference settings; hardware is recorded separately."""
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "weights_sha256": WEIGHTS_SHA256, "version": NEURAL_VERSION,
            "max_length": 256, "window_words": 110, "stride": 55, "title_words": 40}


def _load_inputs(data_dir, cache_dir, out_dir):
    data_dir, cache_dir, out_dir = map(Path, (data_dir, cache_dir, out_dir))
    bundle = FeatureBundle.load(cache_dir / "features.npz")
    if bundle.metadata.get("complete") is not True:
        raise ValueError("Neural experiments require a complete full-corpus feature cache")
    from features import source_fingerprint
    if bundle.metadata.get("fingerprint") != source_fingerprint(data_dir):
        raise ValueError("Feature source files or extraction implementation changed")
    pairs = load_pairs(data_dir)
    for key in ("query_ids", "doc_ids", "object_ids", "labels", "is_train"):
        if not np.array_equal(np.asarray(getattr(bundle, key)), np.asarray(getattr(pairs, key))):
            raise ValueError(f"Feature rows differ from the competition pair table: {key}")
    selection_path = out_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if list(bundle.feature_names) != selection["input_feature_names"]:
        raise ValueError("Feature schema differs from the selected lexical experiment")
    split = ensure_split(bundle, out_dir, selection["seed"])
    if split["sha256"] != selection["split_sha256"]:
        raise ValueError("Lexical and neural experiments have different query splits")
    for part in ("fit", "tune"):
        rows = rows_for(bundle, split["query_ids"][part])
        if dataset_hash(subset(bundle, rows)) != selection[f"{part}_sha256"]:
            raise ValueError(f"{part} data changed after lexical selection")
    return bundle, pairs, selection, split


def aligned_predictions(path, bundle, rows):
    """Join predictions to feature rows by (QueryId, DocumentId), verifying labels.

    Reordering an archive is harmless. Missing/extra/duplicate pairs and label
    mismatches fail explicitly instead of silently associating unrelated scores.
    """
    with np.load(path, allow_pickle=False) as archive:
        required = {"query_ids", "doc_ids", "labels", "selected", "baseline"}
        if not required <= set(archive.files):
            raise ValueError("Prediction archive is missing required arrays")
        arrays = {name: archive[name] for name in required}
    count = len(arrays["labels"])
    if any(value.shape != (count,) for value in arrays.values()):
        raise ValueError("Prediction archive arrays have inconsistent shapes")
    keys = list(zip(arrays["query_ids"].astype(str), arrays["doc_ids"].astype(str), strict=True))
    positions = {key: index for index, key in enumerate(keys)}
    if len(positions) != len(keys):
        raise ValueError("Prediction archive contains duplicate query/document pairs")
    expected = list(zip(np.asarray(bundle.query_ids)[rows].astype(str),
                        np.asarray(bundle.doc_ids)[rows].astype(str), strict=True))
    if len(expected) != len(set(expected)) or set(expected) != set(keys):
        raise ValueError("Prediction archive pairs do not match the requested query partition")
    order = np.asarray([positions[key] for key in expected], dtype=np.int64)
    if not np.array_equal(arrays["labels"][order], np.asarray(bundle.labels)[rows]):
        raise ValueError("Prediction archive labels differ from the feature bundle")
    result = {key: arrays[key][order].astype(np.float64) for key in ("selected", "baseline")}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Prediction archive contains nonfinite lexical scores")
    return result


def _code_hashes():
    return {name: file_hash(ROOT / name) for name in
            ("ranking.py", "experiments.py", "fusion.py", "neural.py", "pipeline.py")}


def _freeze_identity(bundle, selection, split, tune_predictions):
    return {"lexical_selection_sha256": json_hash(selection),
            "split_sha256": split["sha256"],
            "tune_predictions_sha256": file_hash(tune_predictions),
            "feature_source_sha256": json_hash(bundle.metadata["fingerprint"]),
            "implementation_sha256": _code_hashes()}


def _verify_freeze(out_dir, config_dir, identity):
    out_dir, config_dir = Path(out_dir), Path(config_dir)
    freeze = json.loads((out_dir / "fusion-freeze.json").read_text(encoding="utf-8"))
    for key, value in identity.items():
        if freeze.get(key) != value:
            raise ValueError(f"Frozen fusion experiment changed: {key}")
    runtime = json.loads((config_dir / "fusion_config.json").read_text(encoding="utf-8"))
    lexical = json.loads((config_dir / "best_config.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "fusion-selection.json").read_text(encoding="utf-8"))
    if json_hash(runtime) != freeze["fusion_config_sha256"]:
        raise ValueError("Frozen fusion_config.json was modified")
    if json_hash(lexical) != freeze["lexical_selection_sha256"]:
        raise ValueError("Frozen best_config.json differs from lexical selection")
    if json_hash(report) != freeze["fusion_report_sha256"]:
        raise ValueError("Frozen fusion selection report was modified")
    if runtime["neural"] != neural_configuration():
        raise ValueError("Pinned neural model or passage settings changed")
    return freeze, runtime, report


def _score_rows(pairs, cache_dir, model_dir, rows, *, device, batch_size, threads, neural_config):
    settings = {key: neural_config[key] for key in
                ("max_length", "window_words", "stride", "title_words")}
    checkpoint = Path(cache_dir) / f"neural-{device}-b{batch_size}.npz"
    return score_pairs(pairs, Path(cache_dir) / "documents.sqlite", Path(model_dir),
                       checkpoint, np.asarray(rows, dtype=np.int64), device=device,
                       batch_size=batch_size, threads=threads, **settings)


def _save_scores(path, bundle, rows, lexical, baseline, selected, neural, available):
    np.savez_compressed(path, query_ids=np.asarray(bundle.query_ids)[rows].astype(str),
                        doc_ids=np.asarray(bundle.doc_ids)[rows].astype(str),
                        labels=np.asarray(bundle.labels)[rows], lexical=lexical,
                        baseline=baseline, selected=selected, neural=neural, available=available)


def neural_tune(data_dir, cache_dir, out_dir, model_dir, *, device="cpu", batch_size=16,
                threads=4, config_dir=None):
    started = perf_counter()
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    config_dir = ROOT if config_dir is None else Path(config_dir)
    for name in ("audit-lock.json", "final-audit-lock.json", "final-holdout.json"):
        if (out_dir / name).exists():
            raise ValueError("Holdout audit has started; neural tuning is no longer allowed")
    bundle, pairs, selection, split = _load_inputs(data_dir, cache_dir, out_dir)
    rows = rows_for(bundle, split["query_ids"]["tune"])
    predictions_path = out_dir / "tune-predictions.npz"
    predictions = aligned_predictions(predictions_path, bundle, rows)
    tune = subset(bundle, rows)
    metric = NDCG(tune.labels, tune.query_ids, tune.doc_ids)
    if not np.isclose(metric.mean(predictions["selected"]), selection["tune_ndcg"], rtol=0, atol=1e-12):
        raise ValueError("Saved lexical tuning predictions do not reproduce selection nDCG")
    identity = _freeze_identity(bundle, selection, split, predictions_path)
    if (out_dir / "fusion-freeze.json").exists():
        _, _, report = _verify_freeze(out_dir, config_dir, identity)
        print("Fusion tuning is already complete and its frozen inputs match", flush=True)
        return report
    settings = neural_configuration()
    scores, available, neural_metadata = _score_rows(
        pairs, cache_dir, model_dir, rows, device=device, batch_size=batch_size,
        threads=threads, neural_config=settings)
    neural, mask = scores[rows], available[rows]
    if not mask.all() or not np.isfinite(neural).all():
        raise ValueError("All tuning pairs must have neural scores before selecting a fusion")
    result = select_fusion(tune.labels, tune.query_ids, tune.doc_ids,
                           predictions["selected"], neural, mask)
    runtime = {"format_version": FORMAT_VERSION, "config": result["config"], "neural": settings,
               "lexical_selection_sha256": identity["lexical_selection_sha256"],
               "split_sha256": split["sha256"],
               "selection_source": "training tune queries only"}
    final_scores = fuse_scores(tune.query_ids, tune.doc_ids, predictions["selected"],
                               neural, mask, runtime["config"])
    selected_mask = required_rows(tune.query_ids, tune.doc_ids, predictions["selected"], runtime["config"])
    report = {**result, "format_version": FORMAT_VERSION, "metric": METRIC,
              "baseline_kind": "selected lexical ranker",
              "bm25f_ndcg10": metric.mean(predictions["baseline"]),
              "lexical_ndcg10": metric.mean(predictions["selected"]),
              "selected_neural_pairs": int(selected_mask.sum()),
              "selected_neural_queries": len(np.unique(tune.query_ids[selected_mask])),
              "holdout_used_for_selection": False,
              "inference_environment": {"device": device, "batch_size": batch_size, "threads": threads},
              "neural_metadata": neural_metadata, "fusion_config_sha256": json_hash(runtime),
              "lexical_selection_sha256": identity["lexical_selection_sha256"],
              "seconds": round(perf_counter()-started, 3)}
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_dir / "best_config.json", selection)
    write_json(config_dir / "fusion_config.json", runtime)
    write_json(out_dir / "fusion-selection.json", report)
    _save_scores(out_dir / "fusion-tune-predictions.npz", bundle, rows,
                 predictions["selected"], predictions["baseline"], final_scores, neural, mask)
    freeze = {"format_version": FORMAT_VERSION, **identity,
              "fusion_config_sha256": json_hash(runtime), "fusion_report_sha256": json_hash(report),
              "policy": "fixed on tune queries before any final holdout evaluation"}
    write_json(out_dir / "fusion-freeze.json", freeze)
    print(json.dumps({"fusion_selected": runtime["config"], "tune_ndcg10": report["tune_ndcg10"],
                      "lexical_ndcg10": report["lexical_ndcg10"],
                      "neural_pairs": report["selected_neural_pairs"], "holdout_evaluated": False}), flush=True)
    return report


def final_audit(data_dir, cache_dir, out_dir, model_dir, *, device="cpu", batch_size=16,
                threads=4, config_dir=None):
    started = perf_counter()
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    config_dir = ROOT if config_dir is None else Path(config_dir)
    bundle, pairs, selection, split = _load_inputs(data_dir, cache_dir, out_dir)
    identity = _freeze_identity(bundle, selection, split, out_dir / "tune-predictions.npz")
    freeze, runtime, fusion_report = _verify_freeze(out_dir, config_dir, identity)
    lock_path = out_dir / "final-audit-lock.json"
    lock = {"fusion_freeze_sha256": json_hash(freeze),
            "fusion_config_sha256": json_hash(runtime),
            "lexical_selection_sha256": json_hash(selection),
            "policy": "final holdout is report-only; no further tuning"}
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Final audit is locked to a different frozen fusion")
    else:
        # Persist the final neural choice before audit_selection sees holdout labels.
        write_json(lock_path, lock)
    lexical_report = audit_selection(bundle, selection, out_dir, threads)
    rows = rows_for(bundle, split["query_ids"]["holdout"])
    holdout = subset(bundle, rows)
    predictions = aligned_predictions(out_dir / "holdout-predictions.npz", bundle, rows)
    metric = NDCG(holdout.labels, holdout.query_ids, holdout.doc_ids)
    if not np.isclose(metric.mean(predictions["selected"]), lexical_report["selected_ndcg"], rtol=0, atol=1e-12):
        raise ValueError("Saved holdout predictions do not match the lexical audit")
    holdout_hash = dataset_hash(holdout)
    report_path = out_dir / "final-holdout.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("fusion_freeze_sha256") != json_hash(freeze) or report.get("holdout_sha256") != holdout_hash:
            raise ValueError("Saved final holdout report does not match its frozen inputs")
        if not (out_dir / "final-holdout-predictions.npz").is_file():
            raise ValueError("Saved final audit is missing its prediction artifact")
        return report
    needed = required_rows(holdout.query_ids, holdout.doc_ids,
                           predictions["selected"], runtime["config"])
    neural = np.full(len(rows), np.nan, dtype=np.float32)
    available = np.zeros(len(rows), dtype=bool)
    neural_metadata = None
    if needed.any():
        scores, mask, neural_metadata = _score_rows(
            pairs, cache_dir, model_dir, rows[needed], device=device, batch_size=batch_size,
            threads=threads, neural_config=runtime["neural"])
        neural, available = scores[rows], mask[rows]
    selected = fuse_scores(holdout.query_ids, holdout.doc_ids, predictions["selected"],
                           neural, available, runtime["config"])
    selected_per_query = metric.per_query(selected)
    lexical_per_query = metric.per_query(predictions["selected"])
    baseline_per_query = metric.per_query(predictions["baseline"])
    report = {"format_version": FORMAT_VERSION, "metric": METRIC,
              "fusion_freeze_sha256": json_hash(freeze), "fusion_config_sha256": json_hash(runtime),
              "lexical_selection_sha256": json_hash(selection), "split_sha256": split["sha256"],
              "holdout_sha256": holdout_hash, "fusion_config": runtime["config"],
              "selected_ndcg": float(selected_per_query.mean()),
              "lexical_ndcg": float(lexical_per_query.mean()),
              "bm25f_ndcg": float(baseline_per_query.mean()),
              "vs_bm25f": paired_bootstrap(selected_per_query, baseline_per_query, selection["seed"]+1),
              "vs_lexical": paired_bootstrap(selected_per_query, lexical_per_query, selection["seed"]+2),
              "queries": len(metric.query_ids), "pairs": len(rows),
              "neural_pairs": int(needed.sum()),
              "neural_queries": len(np.unique(holdout.query_ids[needed])),
              "used_for_selection": False,
              "per_query_selected": dict(zip(metric.query_ids.tolist(), selected_per_query.tolist())),
              "per_query_lexical": dict(zip(metric.query_ids.tolist(), lexical_per_query.tolist())),
              "per_query_bm25f": dict(zip(metric.query_ids.tolist(), baseline_per_query.tolist())),
              "inference_environment": {"device": device, "batch_size": batch_size, "threads": threads},
              "neural_metadata": neural_metadata, "seconds": round(perf_counter()-started, 3)}
    _save_scores(out_dir / "final-holdout-predictions.npz", bundle, rows,
                 predictions["selected"], predictions["baseline"], selected, neural, available)
    write_json(report_path, report)
    print(json.dumps({key: report[key] for key in
                      ("selected_ndcg", "lexical_ndcg", "bm25f_ndcg", "neural_pairs", "seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("neural-tune", "audit"))
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_dir", type=Path,
                        default=ROOT.parents[1] / "models" / "mmarco-miniLM")
    parser.add_argument("--config_dir", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch_size and threads must be positive")
    stage = neural_tune if args.stage == "neural-tune" else final_audit
    try:
        stage(args.data_dir, args.cache_dir, args.output_dir, args.model_dir,
              device=args.device, batch_size=args.batch_size, threads=args.threads,
              config_dir=args.config_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
