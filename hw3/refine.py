#!/usr/bin/env python3
"""Extend the predefined search using the same fit/tuning queries only.

No test table or holdout scores enter this script. The saved configuration is
ready for a separate, subsequent audit and refit on all training queries.
Completed candidates can be resumed only with matching data, code, versions,
split, and configurations; their native model and predictions are checksummed.
"""

import argparse
from itertools import combinations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
from time import perf_counter
import zipfile

import numpy as np

from data import load_train
from experiments import make_split
from features import normalize_scores
from metrics import mean_ndcg
from modeling import fit_component, predict_component, save_models, subset
from solution import digest

ROOT = Path(__file__).resolve().parent


def candidates(gain="exp2", seed=42):
    """Eleven fixed additions; no parameters depend on validation outcomes."""
    metric = "NDCG:top=10;type=" + ("Exp" if gain == "exp2" else "Base") + ";denominator=LogPosition"
    standard = {"depth": 6, "learning_rate": .04, "l2_leaf_reg": 5., "eval_metric": metric}

    def cat(name, *, mode="rank", random_seed=seed, **params):
        return {"name": name, "family": "catboost", "features": mode,
                "iterations": 1200, "gain": gain, "seed": random_seed,
                "params": dict(standard, **params)}

    result = [cat(f"cat_yeti_rank_d{depth}", depth=depth) for depth in (4, 5, 7, 8)]
    result += [cat("cat_yeti_relative_d6", mode="relative"),
               cat("cat_yeti_rank_regularized", l2_leaf_reg=20., bootstrap_type="Bayesian",
                   bagging_temperature=.3, random_strength=.5),
               cat("cat_pairlogit_rank_d6", loss_function="PairLogit")]
    for objective in ("lambdarank", "rank_xendcg"):
        result.append({"name": f"lgb_{objective}_relative_15", "family": "lightgbm",
                       "features": "relative", "iterations": 1200, "gain": gain, "seed": seed,
                       "params": {"objective": objective, "num_leaves": 15,
                                  "min_data_in_leaf": 80, "lambda_l2": 10.}})
    result += [cat("cat_yeti_rank_seed17", random_seed=17),
               cat("cat_yeti_rank_seed73", random_seed=73)]
    return result


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_metadata(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("metadata.json"))


def read_predictions(path, rows):
    with np.load(path, allow_pickle=False) as archive:
        predictions = np.asarray(archive["predictions"], dtype=np.float64)
    if predictions.shape != (rows,) or not np.isfinite(predictions).all():
        raise ValueError(f"Invalid cached predictions: {path}")
    return predictions


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(f"Unsafe candidate name: {name!r}")
    return name


def score(tune, prediction, gain):
    return mean_ndcg(tune.labels, prediction, tune.query_ids, gain=gain)


def read_initial(initial_dir, train_hash, split, feature_names, tune, gain, seed):
    # Whitelist search metadata and candidate records. In particular, no
    # selection/holdout field from the original report is accessed or returned.
    with (initial_dir / "experiments.json").open(encoding="utf-8") as handle:
        full_report = json.load(handle)
    report = {key: full_report[key] for key in
              ("train_sha256", "seed", "gain", "k", "zero_idcg", "ties",
               "quick_smoke_only", "split_queries", "candidates")}
    del full_report
    if (report["quick_smoke_only"] or report["train_sha256"] != train_hash
            or report["seed"] != seed or report["gain"] != gain or report["k"] != 10
            or report["zero_idcg"] != 0 or report["ties"] != "original train row order"):
        raise ValueError("Initial results use different data/split/metric or are a quick smoke run")
    if report["split_queries"] != {key: value.tolist() for key, value in split.items()}:
        raise ValueError("Initial query membership/order does not match this run")
    expected_names = {"baseline_yetirank200", "lgb_raw_31", "lgb_rank_31", "lgb_relative_31",
                      "lgb_raw_63", "lgb_relative_63", "lgb_xendcg_relative", "cat_yeti_raw",
                      "cat_yeti_rank", "cat_ndcg_rank"}
    if {record["name"] for record in report["candidates"]} != expected_names:
        raise ValueError("Initial search must contain all ten completed official candidates")
    cases, predictions, provenance = {}, {}, []
    for record in report["candidates"]:
        name = safe_name(record["name"])
        if name in cases:
            raise ValueError(f"Duplicate initial candidate: {name}")
        model_path = initial_dir / "models" / f"{name}.bin"
        prediction_path = initial_dir / "models" / f"{name}.npz"
        config = record["config"]
        metadata = read_metadata(model_path)
        if (metadata["feature_names"] != list(feature_names)
                or metadata["config"]["components"] != [config]
                or metadata["training"].get("purpose") != "tuning"
                or metadata["training"].get("fit_queries") != len(split["fit"])):
            raise ValueError(f"Initial model metadata mismatch: {name}")
        prediction = read_predictions(prediction_path, len(tune.query_ids))
        recomputed = score(tune, prediction, gain)
        if not np.isclose(recomputed, record["tuning_ndcg"], rtol=0, atol=1e-12):
            raise ValueError(f"Initial prediction order/score mismatch: {name}")
        cases[name], predictions[name] = config, prediction
        provenance.append({"name": name, "config": config, "tuning_ndcg": recomputed,
                           "model_sha256": digest(model_path),
                           "predictions_sha256": digest(prediction_path)})
    return cases, predictions, provenance


def choose(cases, predictions, tune, gain="exp2", seed=42):
    """Choose singles, top-four pairwise blends, and one fixed three-seed bag."""
    choices = []
    singles = sorted(((score(tune, predictions[name], gain), name)
                      for name in cases), key=lambda item: -item[0])

    def add(name, names, weights, normalization):
        result = np.zeros(len(tune.query_ids), dtype=np.float64)
        for component_name, weight in zip(names, weights):
            result += weight * normalize_scores(predictions[component_name], tune.query_ids, normalization)
        proposal = {"seed": seed, "score_normalization": normalization,
                    "components": [dict(cases[n], weight=w) for n, w in zip(names, weights)]}
        choices.append((score(tune, result, gain), name, proposal))

    # Singles first means exact ties prefer a single model over a blend.
    for _, name in singles:
        add(name, [name], [1.], "none")
    for a, b in combinations([name for _, name in singles[:4]], 2):
        for weight in (.25, .5, .75):
            add(f"{a}+{b}:zscore:{weight}", [a, b], [weight, 1 - weight], "zscore")
    seed_names = ["cat_yeti_rank", "cat_yeti_rank_seed17", "cat_yeti_rank_seed73"]
    if all(name in cases for name in seed_names):
        add("cat_yeti_rank:three_seeds", seed_names, [1 / 3] * 3, "zscore")
    best = max(choices, key=lambda item: item[0])
    return best, [{"name": name, "tuning_ndcg": value} for value, name, _ in choices]


def run(data_dir, initial_dir, output_dir, *, gain="exp2", seed=42, threads=4,
        specifications=None):
    started = perf_counter()
    data_dir, initial_dir, output_dir = map(lambda p: Path(p).resolve(),
                                          (data_dir, initial_dir, output_dir))
    if output_dir in {data_dir, initial_dir}:
        raise ValueError("Use a distinct output directory for refinement")
    if threads < 1:
        raise ValueError("threads must be positive")
    data = load_train(data_dir / "train.csv")
    split = make_split(data.query_ids, seed)
    fit = subset(data, np.flatnonzero(np.isin(data.query_ids, split["fit"])))
    tune = subset(data, np.flatnonzero(np.isin(data.query_ids, split["tuning"])))
    feature_names = data.feature_names
    del data  # The holdout is never materialized as a Dataset or evaluated.
    train_hash = digest(data_dir / "train.csv")
    cases, predictions, initial = read_initial(initial_dir, train_hash, split, feature_names, tune, gain, seed)
    specs = candidates(gain, seed) if specifications is None else specifications
    if len({safe_name(item["name"]) for item in specs}) != len(specs):
        raise ValueError("Duplicate refinement candidate names")
    if any(item["name"] in cases for item in specs):
        raise ValueError("Refinement candidate names overlap the initial search")
    plan = {"train_sha256": train_hash, "gain": gain, "seed": seed, "threads": threads,
            "split_queries": {key: value.tolist() for key, value in split.items()},
            "fit_rows": len(fit.query_ids), "tuning_rows": len(tune.query_ids),
            "feature_names": list(feature_names), "candidates": specs,
            "initial_candidates": initial,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("numpy", "pandas", "catboost", "lightgbm")},
            "source_sha256": {name: digest(ROOT / name) for name in
                              ("refine.py", "modeling.py", "features.py", "data.py", "metrics.py",
                               "experiments.py", "solution.py", "requirements.txt")},
            "selection_rule": "Singles + zscore .25/.5/.75 blends of top 4 singles + fixed 3-seed bag",
            "holdout_used": False, "test_read": False}
    signature = canonical_hash(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature or previous.get("plan") != plan:
            raise ValueError("Resume inputs/configuration/code changed; use a fresh output directory")
    else:
        atomic_json(plan_path, {"signature": signature, "plan": plan})
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    records = []
    for spec in specs:
        name = spec["name"]
        model_path, prediction_path = model_dir / f"{name}.bin", model_dir / f"{name}.npz"
        record_path = model_dir / f"{name}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (record.get("signature") != signature or record.get("specification") != spec
                    or record.get("model_sha256") != digest(model_path)
                    or record.get("predictions_sha256") != digest(prediction_path)):
                raise ValueError(f"Corrupt or incompatible resume artifacts: {name}")
            prediction = read_predictions(prediction_path, len(tune.query_ids))
            metadata = read_metadata(model_path)
            if (metadata["config"]["components"] != [record["config"]]
                    or metadata["training"].get("signature") != signature
                    or metadata["feature_names"] != list(feature_names)
                    or not np.isclose(score(tune, prediction, gain), record["tuning_ndcg"], rtol=0, atol=1e-12)):
                raise ValueError(f"Resume model or prediction identity mismatch: {name}")
            resumed = True
        else:
            begin = perf_counter()
            model, config = fit_component(fit, spec, tune, seed=spec.get("seed", seed), threads=threads)
            prediction = predict_component(model, tune, config, threads)
            # Write the record last: incomplete runs retrain this candidate.
            with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=model_dir) as temporary:
                tmp = Path(temporary)
                save_models(tmp / "model.bin", [model],
                            {"seed": spec.get("seed", seed), "score_normalization": "none", "components": [config]},
                            feature_names, {"purpose": "tuning_refinement", "signature": signature,
                                            "train_sha256": train_hash, "fit_queries": len(split["fit"])})
                np.savez_compressed(tmp / "predictions.npz", predictions=prediction)
                os.replace(tmp / "model.bin", model_path)
                os.replace(tmp / "predictions.npz", prediction_path)
            record = {"signature": signature, "name": name, "specification": spec, "config": config,
                      "seconds": round(perf_counter() - begin, 3), "tuning_ndcg": score(tune, prediction, gain),
                      "tuning_ndcg_other_gain": score(tune, prediction, "linear" if gain == "exp2" else "exp2"),
                      "model_sha256": digest(model_path), "predictions_sha256": digest(prediction_path)}
            atomic_json(record_path, record)
            del model
            resumed = False
        cases[name], predictions[name] = record["config"], prediction
        records.append(record)
        report = {"signature": signature, "train_sha256": train_hash, "gain": gain, "seed": seed,
                  "initial_candidates": initial, "candidates": records, "holdout_used": False, "test_read": False}
        atomic_json(output_dir / "experiments.json", report)
        print(json.dumps({"name": name, "resumed": resumed, "seconds": record["seconds"],
                          "iterations": record["config"]["iterations"], "tuning_ndcg": record["tuning_ndcg"]}), flush=True)
    (best_score, name, best_config), choices = choose(cases, predictions, tune, gain, seed)
    best_config["status"] = "selected_on_training_queries"
    best_config["selection"] = {"name": name, "tuning_ndcg": best_score, "gain": gain,
                                "refinement_signature": signature, "holdout_used": False}
    atomic_json(output_dir / "best_config.json", best_config)
    report["selection"] = {"name": name, "tuning_ndcg": best_score, "choices": choices}
    report["seconds_this_invocation"] = round(perf_counter() - started, 3)
    atomic_json(output_dir / "experiments.json", report)
    print(json.dumps({"selected": name, "tuning_ndcg": best_score,
                      "seconds_this_invocation": report["seconds_this_invocation"], "holdout_used": False}), flush=True)
    return best_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--initial_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gain", choices=("exp2", "linear"), default="exp2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        run(args.data_dir, args.initial_dir, args.output_dir, gain=args.gain, seed=args.seed, threads=args.threads)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
