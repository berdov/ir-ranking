#!/usr/bin/env python3
"""Train-only model selection, a frozen holdout audit and final refitting.

The stages are deliberately separate: tuning writes its decision before the
holdout is evaluated. Neural post-blend choices can therefore be fixed using
the saved tuning predictions before ``audit`` is called. Public Kaggle scores
are not input to any stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ranking import (METRIC, NDCG, environment_versions, fit_component, json_hash,
                     load_component, normalize_scores, predict_component,
                     save_component, subset, validate_bundle, write_json)


FORMAT_VERSION = 1


def implementation_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("ranking.py", "experiments.py")}


def make_split(query_ids, seed=42):
    unique = np.unique(np.asarray(query_ids).astype(str))
    if len(unique) < 7:
        raise ValueError("At least seven labelled queries are required")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_eval = min(2000, max(1, int(len(unique) * .15)))
    return {"fit": sorted(shuffled[2 * n_eval:].tolist()),
            "tune": sorted(shuffled[n_eval:2 * n_eval].tolist()),
            "holdout": sorted(shuffled[:n_eval].tolist())}


def split_manifest(bundle, seed=42):
    train = np.asarray(bundle.is_train, dtype=bool)
    ids = np.asarray(bundle.query_ids).astype(str)
    queries = make_split(ids[train], seed)
    payload = {"seed": int(seed), "query_ids": queries,
               "training_queries_sha256": json_hash(sorted(set(ids[train])))}
    return dict(payload, sha256=json_hash(payload),
                query_counts={name: len(value) for name, value in queries.items()},
                row_counts={name: int(np.isin(ids, value).sum()) for name, value in queries.items()})


def ensure_split(bundle, directory, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = split_manifest(bundle, seed)
    path = directory / "splits.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("Existing query split differs; do not silently replace a validation split")
    else:
        write_json(path, expected)
    return expected


def rows_for(bundle, query_ids):
    return np.flatnonzero(np.asarray(bundle.is_train, dtype=bool)
                          & np.isin(np.asarray(bundle.query_ids).astype(str), query_ids))


def dataset_hash(bundle):
    """Fingerprint a labelled subset, never called on holdout during selection."""
    digest = hashlib.sha256()
    digest.update(json_hash(list(bundle.feature_names)).encode())
    for key in ("X", "labels", "query_ids", "doc_ids"):
        array = np.ascontiguousarray(getattr(bundle, key))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def configurations(feature_sets, seed=42, quick=False):
    output = []
    for set_name, columns in feature_sets.items():
        modes = ("raw", "rank") if set_name == "lexical" else ("rank",)
        for family in ("lightgbm", "catboost"):
            for mode in modes:
                output.append({"name": f"{set_name}_{family}_{mode}", "feature_set": set_name,
                               "family": family, "feature_mode": mode, "columns": list(columns),
                               "iterations": 8 if quick else (1200 if family == "lightgbm" else 1000),
                               "seed": int(seed), "params": {}})
    return output


def _feature_sets(bundle, feature_sets):
    if feature_sets is None:
        names = list(bundle.feature_names)
        lexical = [name for name in names if not name.startswith(("neural_", "crossencoder_"))]
        feature_sets = {"lexical": lexical}
        if lexical != names:
            feature_sets["augmented"] = names
    if not feature_sets or any(not columns for columns in feature_sets.values()):
        raise ValueError("Every feature set must contain at least one feature")
    known = set(bundle.feature_names)
    for name, columns in feature_sets.items():
        if not name or not name.replace("_", "").isalnum():
            raise ValueError("Feature set names must contain only letters, digits or underscores")
        if len(set(columns)) != len(columns) or not set(columns) <= known:
            raise ValueError("Feature set contains missing or duplicate columns")
    return feature_sets


def _prediction_archive(path, data, **predictions):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, query_ids=np.asarray(data.query_ids).astype(str),
                        doc_ids=np.asarray(data.doc_ids).astype(str),
                        labels=np.asarray(data.labels), **predictions)


def select_from_tuning(cases, predictions, tune, baseline_name):
    """Only this tuning subset is visible to the configuration chooser."""
    metric = NDCG(tune.labels, tune.query_ids, tune.doc_ids)
    choices = []
    for name, config in cases.items():
        proposal = {"components": [dict(config, weight=1.)], "score_normalization": "none"}
        choices.append((metric.mean(predictions[name]), name, proposal, predictions[name]))
    best_family = {}
    for score, name, proposal, _ in choices:
        component = proposal["components"][0]
        family = component["family"]
        if family != "column" and (family not in best_family or score > best_family[family][0]):
            best_family[family] = (score, name, component)
    if set(best_family) == {"catboost", "lightgbm"}:
        _, cat_name, cat = best_family["catboost"]
        _, lgb_name, lgb = best_family["lightgbm"]
        C = normalize_scores(predictions[cat_name], tune.query_ids, "zscore")
        L = normalize_scores(predictions[lgb_name], tune.query_ids, "zscore")
        for cat_weight in (.25, .5, .75):
            scores = cat_weight * C + (1 - cat_weight) * L
            name = f"blend_{cat_name}_{lgb_name}_cat{cat_weight:g}"
            proposal = {"components": [dict(cat, weight=cat_weight), dict(lgb, weight=1-cat_weight)],
                        "score_normalization": "zscore"}
            choices.append((metric.mean(scores), name, proposal, scores))
    # Stable max leaves the baseline/simpler candidate first on exact ties.
    chosen = max(choices, key=lambda item: item[0])
    comparison = [{"name": name, "tune_ndcg": value,
                   "delta_vs_bm25f": value - metric.mean(predictions[baseline_name])}
                  for value, name, _, _ in choices]
    return chosen, comparison


def train_and_select(bundle, out_dir, threads=4, seed=42, feature_sets=None, quick=False):
    """Compare fixed candidates on tune only; save selected tuning predictions.

    A matching checkpoint allows interrupted tuning to resume. The checkpoint
    signature includes all fit/tune features and labels plus the candidate
    configuration. It never hashes or learns from holdout labels.
    """
    started = perf_counter()
    validate_bundle(bundle)
    out_dir = Path(out_dir)
    split = ensure_split(bundle, out_dir, seed)
    if (out_dir / "audit-lock.json").exists():
        raise ValueError("Holdout audit already started in this directory; selection is frozen")
    fit = subset(bundle, rows_for(bundle, split["query_ids"]["fit"]))
    tune = subset(bundle, rows_for(bundle, split["query_ids"]["tune"]))
    fit_hash, tune_hash = dataset_hash(fit), dataset_hash(tune)
    implementation = implementation_hashes()
    environment = environment_versions()
    metric = NDCG(tune.labels, tune.query_ids, tune.doc_ids)
    baseline_columns = [name for name in bundle.feature_names if name.startswith("bm25f_")]
    if not baseline_columns:
        raise ValueError("At least one bm25f_* feature is required for the baseline")
    baseline_scores = {name: metric.mean(tune.X[:, bundle.feature_names.index(name)])
                       for name in baseline_columns}
    baseline_column = max(baseline_columns, key=lambda name: baseline_scores[name])
    baseline_name = "baseline_bm25f"
    baseline = {"name": baseline_name, "family": "column", "column": baseline_column}
    baseline_prediction = predict_component(None, tune, baseline)
    cases = {baseline_name: baseline}
    predictions = {baseline_name: baseline_prediction}
    report = {"format_version": FORMAT_VERSION, "metric": METRIC, "seed": int(seed),
              "split_sha256": split["sha256"], "query_counts": split["query_counts"],
              "fit_sha256": fit_hash, "tune_sha256": tune_hash,
              "quick_smoke_only": bool(quick), "environment": environment,
              "implementation_sha256": implementation,
              "baseline_search": [{"column": name, "tune_ndcg": value}
                                  for name, value in baseline_scores.items()],
              "selected_baseline": {"column": baseline_column,
                                    "tune_ndcg": baseline_scores[baseline_column]},
              "candidates": [], "holdout_used_for_selection": False}
    write_json(out_dir / "experiments.json", report)
    model_dir = out_dir / "tuning-models"
    model_dir.mkdir(exist_ok=True)
    for candidate in configurations(_feature_sets(bundle, feature_sets), seed, quick):
        begin = perf_counter()
        name = candidate["name"]
        signature = json_hash({"config": candidate, "fit": fit_hash, "tune": tune_hash,
                               "schema_version": FORMAT_VERSION,
                               "implementation_sha256": implementation,
                               "environment": environment})
        checkpoint_path = model_dir / f"{name}.json"
        predictions_path = model_dir / f"{name}-predictions.npz"
        previous = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
        if previous and previous.get("signature") == signature and predictions_path.exists():
            with np.load(predictions_path, allow_pickle=False) as archive:
                prediction = archive["scores"]
            config = previous["config"]
            result = previous["result"]
            metric.mean(prediction)
            print(json.dumps({"resumed": name, "tune_ndcg": result["tune_ndcg"]}), flush=True)
        else:
            print(json.dumps({"training": name, "max_iterations": candidate["iterations"]}), flush=True)
            model, config = fit_component(fit, candidate, tune, threads)
            prediction = predict_component(model, tune, config, threads)
            filename = save_component(model, config, model_dir, name)
            del model
            result = {"name": name, "config": config, "seconds": round(perf_counter()-begin, 3),
                      "tune_ndcg": metric.mean(prediction), "model_file": filename,
                      "per_query_tune": dict(zip(metric.query_ids.tolist(), metric.per_query(prediction).tolist()))}
            np.savez_compressed(predictions_path, scores=prediction)
            write_json(checkpoint_path, {"signature": signature, "config": config, "result": result})
            print(json.dumps({key: result[key] for key in ("name", "seconds", "tune_ndcg")}), flush=True)
        cases[name], predictions[name] = config, prediction
        report["candidates"].append(result)
        write_json(out_dir / "experiments.json", report)
    (best_score, name, proposal, selected_scores), choices = select_from_tuning(cases, predictions, tune, baseline_name)
    selection = {"format_version": FORMAT_VERSION, "seed": int(seed), "name": name,
                 "tune_ndcg": best_score, "split_sha256": split["sha256"],
                 "fit_sha256": fit_hash, "tune_sha256": tune_hash,
                 "baseline": baseline, "input_feature_names": list(bundle.feature_names),
                 "quick_smoke_only": bool(quick), "selection_source": "training tune queries only",
                 **proposal}
    # This decision is committed before the holdout audit stage exists.
    write_json(out_dir / "selection.json", selection)
    _prediction_archive(out_dir / "tune-predictions.npz", tune,
                        selected=selected_scores, baseline=baseline_prediction)
    report.update(selection=selection, choices=choices, seconds=round(perf_counter()-started, 3),
                  per_query_selected=dict(zip(metric.query_ids.tolist(), metric.per_query(selected_scores).tolist())),
                  per_query_baseline=dict(zip(metric.query_ids.tolist(), metric.per_query(baseline_prediction).tolist())))
    write_json(out_dir / "experiments.json", report)
    lines = ["| Variant | Tune nDCG@10 | Difference vs BM25F |", "| --- | ---: | ---: |"]
    lines.extend(f"| {c['name']} | {c['tune_ndcg']:.6f} | {c['delta_vs_bm25f']:+.6f} |" for c in choices)
    (out_dir / "tuning-table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": name, "tune_ndcg": best_score,
                      "holdout_evaluated": False, "selection": str(out_dir / "selection.json")}), flush=True)
    return selection


def _verify_selection(bundle, selection):
    if selection.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unknown selection format")
    if list(bundle.feature_names) != selection["input_feature_names"]:
        raise ValueError("Feature schema differs from the selected experiment")
    split = split_manifest(bundle, selection["seed"])
    if split["sha256"] != selection["split_sha256"]:
        raise ValueError("Training queries differ from the selected experiment")
    return split


def fit_predict_selection(train, target, selection, threads=4, model_dir=None):
    predictions = np.zeros(len(target.query_ids), dtype=np.float64)
    total_weight = 0.
    files = []
    for i, component in enumerate(selection["components"]):
        weight = float(component.get("weight", 1.))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Ensemble weights must be finite and nonnegative")
        model, config = fit_component(train, component, threads=threads)
        values = predict_component(model, target, config, threads)
        predictions += weight * normalize_scores(values, target.query_ids, selection["score_normalization"])
        total_weight += weight
        if model_dir is not None:
            files.append(save_component(model, config, model_dir, f"component_{i}"))
        del model
    if total_weight <= 0:
        raise ValueError("Ensemble needs a positive total weight")
    return predictions / total_weight, files


def paired_bootstrap(first, second, seed=43, repeats=2000):
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Bootstrap requires finite aligned per-query values")
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {"mean_difference": float(delta.mean()),
            "paired_bootstrap_95pct_ci": np.quantile(means, [.025, .975]).tolist(),
            "bootstrap_repeats": int(repeats), "bootstrap_seed": int(seed)}


def audit_selection(bundle, selection, out_dir, threads=4):
    """Evaluate a frozen selection once, with fit+tune queries used for refitting."""
    started = perf_counter()
    validate_bundle(bundle)
    split = _verify_selection(bundle, selection)
    out_dir = Path(out_dir)
    ensure_split(bundle, out_dir, selection["seed"])
    selection_hash = json_hash(selection)
    lock = out_dir / "audit-lock.json"
    cached_report = None
    if lock.exists():
        previous = json.loads(lock.read_text())
        if previous["selection_sha256"] != selection_hash:
            raise ValueError("Holdout is frozen for a different configuration")
        if (out_dir / "holdout.json").exists():
            cached_report = json.loads((out_dir / "holdout.json").read_text())
            if cached_report["selection_sha256"] != selection_hash:
                raise ValueError("Saved holdout report does not match its audit lock")
    else:
        write_json(lock, {"selection_sha256": selection_hash, "selection": selection,
                          "policy": "holdout is report-only; do not revise selection from its result"})
    fit_rows = rows_for(bundle, split["query_ids"]["fit"])
    tune_rows = rows_for(bundle, split["query_ids"]["tune"])
    # Catch accidental changes between selection and audit without inspecting the
    # holdout labels before the decision is locked.
    if dataset_hash(subset(bundle, fit_rows)) != selection["fit_sha256"]:
        raise ValueError("Fit features or labels changed after selection")
    if dataset_hash(subset(bundle, tune_rows)) != selection["tune_sha256"]:
        raise ValueError("Tune features or labels changed after selection")
    train = subset(bundle, np.sort(np.r_[fit_rows, tune_rows]))
    holdout = subset(bundle, rows_for(bundle, split["query_ids"]["holdout"]))
    holdout_hash = dataset_hash(holdout)
    if cached_report is not None:
        if cached_report.get("holdout_sha256") != holdout_hash:
            raise ValueError("Holdout data changed after its recorded evaluation")
        return cached_report
    prediction, _ = fit_predict_selection(train, holdout, selection, threads,
                                         out_dir / "audit-models")
    baseline = predict_component(None, holdout, selection["baseline"])
    metric = NDCG(holdout.labels, holdout.query_ids, holdout.doc_ids)
    selected_per_query, baseline_per_query = metric.per_query(prediction), metric.per_query(baseline)
    report = {"selection_sha256": selection_hash, "split_sha256": split["sha256"],
              "holdout_sha256": holdout_hash,
              "selected_name": selection["name"], "metric": METRIC,
              "selected_ndcg": float(selected_per_query.mean()),
              "baseline_ndcg": float(baseline_per_query.mean()),
              "queries": len(metric.query_ids), "pairs": len(prediction),
              "refit_queries": len(set(train.query_ids)), "used_for_selection": False,
              "per_query_selected": dict(zip(metric.query_ids.tolist(), selected_per_query.tolist())),
              "per_query_baseline": dict(zip(metric.query_ids.tolist(), baseline_per_query.tolist())),
              "seconds": round(perf_counter()-started, 3),
              **paired_bootstrap(selected_per_query, baseline_per_query, selection["seed"] + 1)}
    _prediction_archive(out_dir / "holdout-predictions.npz", holdout, selected=prediction, baseline=baseline)
    write_json(out_dir / "holdout.json", report)
    print(json.dumps({key: report[key] for key in ("selected_ndcg", "baseline_ndcg", "mean_difference", "seconds")}), flush=True)
    return report


def fit_final(bundle, selection, model_dir, threads=4, force=False):
    """Refit the frozen selection on every labelled query, using fixed iterations."""
    started = perf_counter()
    validate_bundle(bundle)
    split = _verify_selection(bundle, selection)
    fit_rows = rows_for(bundle, split["query_ids"]["fit"])
    tune_rows = rows_for(bundle, split["query_ids"]["tune"])
    if dataset_hash(subset(bundle, fit_rows)) != selection["fit_sha256"]:
        raise ValueError("Fit features or labels changed after selection")
    if dataset_hash(subset(bundle, tune_rows)) != selection["tune_sha256"]:
        raise ValueError("Tune features or labels changed after selection")
    train = subset(bundle, np.flatnonzero(bundle.is_train))
    train_hash = dataset_hash(train)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "metadata.json"
    if path.exists() and not force:
        previous = json.loads(path.read_text())
        if previous["training_sha256"] == train_hash and previous["selection_sha256"] == json_hash(selection):
            if all(name is None or (model_dir / name).is_file() for name in previous["model_files"]):
                return previous
        raise ValueError("Model directory contains another run; use another directory or force=True")
    # Avoid spending time predicting the complete training set during refitting.
    filenames = []
    for index, component in enumerate(selection["components"]):
        print(json.dumps({"final_refit": component["name"], "iterations": component.get("iterations")}), flush=True)
        model, config = fit_component(train, component, threads=threads)
        stem = f"component_{index}_{json_hash(selection)[:12]}_{train_hash[:12]}"
        filenames.append(save_component(model, config, model_dir, stem))
        del model
    metadata = {"format_version": FORMAT_VERSION, "selection": selection,
                "selection_sha256": json_hash(selection), "training_sha256": train_hash,
                "input_feature_names": list(bundle.feature_names), "model_files": filenames,
                "training_queries": len(set(train.query_ids)), "training_rows": len(train.labels),
                "environment": environment_versions(), "device": "CPU", "threads": int(threads),
                "seconds": round(perf_counter()-started, 3)}
    write_json(path, metadata)
    return metadata


def predict(bundle, model_dir, threads=4):
    """Return one score per bundle row, preserving its original order."""
    path = Path(model_dir)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported model archive version")
    if metadata["input_feature_names"] != list(bundle.feature_names):
        raise ValueError("Prediction feature schema differs from the trained model")
    selection = metadata["selection"]
    if len(metadata["model_files"]) != len(selection["components"]):
        raise ValueError("Model manifest has an inconsistent number of components")
    output = np.zeros(len(bundle.query_ids), dtype=np.float64)
    total_weight = 0.
    for component, filename in zip(selection["components"], metadata["model_files"], strict=True):
        model = load_component(path, component, filename)
        values = predict_component(model, bundle, component, threads)
        weight = float(component.get("weight", 1.))
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("Invalid saved ensemble weight")
        output += weight * normalize_scores(values, bundle.query_ids, selection["score_normalization"])
        total_weight += weight
        del model
    if total_weight <= 0:
        raise ValueError("Invalid saved total ensemble weight")
    return output / total_weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "audit", "fit", "predict"))
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Synthetic smoke test only, eight trees per candidate")
    args = parser.parse_args()
    from data import FeatureBundle
    bundle = FeatureBundle.load(args.features)
    if args.stage == "tune":
        train_and_select(bundle, args.output_dir, args.threads, args.seed, quick=args.quick)
        return
    selected_path = args.selection or args.output_dir / "selection.json"
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    model_dir = args.model_dir or args.output_dir / "final-models"
    if args.stage == "audit":
        audit_selection(bundle, selection, args.output_dir, args.threads)
    elif args.stage == "fit":
        fit_final(bundle, selection, model_dir, args.threads)
    else:
        scores = predict(bundle, model_dir, args.threads)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_dir / "predictions.npz", scores=scores,
                            query_ids=bundle.query_ids, doc_ids=bundle.doc_ids,
                            object_ids=bundle.object_ids, is_train=bundle.is_train)


if __name__ == "__main__":
    main()
