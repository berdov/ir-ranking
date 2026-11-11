#!/usr/bin/env python3
"""Evaluate an already frozen configuration on held-out training queries."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from data import load_train
from experiments import make_split, write_json
from features import normalize_scores
from metrics import per_query_ndcg
from modeling import fit_component, predict_component, subset
from solution import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--config_file", type=Path, required=True)
    parser.add_argument("--report_file", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    started = perf_counter()
    config_hash = digest(args.config_file)
    config = json.loads(args.config_file.read_text())
    seed = config["seed"]
    data = load_train(args.data_dir / "train.csv")
    split = make_split(data.query_ids, seed)
    is_holdout = np.isin(data.query_ids, split["holdout"])
    train = subset(data, np.flatnonzero(~is_holdout))
    holdout = subset(data, np.flatnonzero(is_holdout))
    baseline = {"family": "catboost", "features": "raw", "iterations": 200,
                "params": {}, "gain": "exp2"}
    model, _ = fit_component(train, baseline, seed=0, threads=args.threads)
    baseline_scores = predict_component(model, holdout, baseline, args.threads)
    del model
    scores = np.zeros(len(holdout.query_ids), dtype=np.float64)
    total_weight = 0.
    for component in config["components"]:
        model, _ = fit_component(train, component, seed=component.get("seed", seed),
                                  threads=args.threads)
        prediction = predict_component(model, holdout, component, args.threads)
        weight = component.get("weight", 1.)
        scores += weight * normalize_scores(prediction, holdout.query_ids,
                                               config.get("score_normalization", "none"))
        total_weight += weight
        del model
    scores /= total_weight
    report = {"purpose": "audit of a frozen configuration; does not modify selection",
              "train_sha256": digest(args.data_dir / "train.csv"),
              "config_sha256": config_hash, "seed": seed, "threads": args.threads,
              "fit_queries": len(np.unique(train.query_ids)),
              "holdout_queries": len(split["holdout"]), "used_for_selection": False}
    for gain in ("exp2", "linear"):
        _, base = per_query_ndcg(holdout.labels, baseline_scores, holdout.query_ids, gain=gain)
        _, selected = per_query_ndcg(holdout.labels, scores, holdout.query_ids, gain=gain)
        difference = selected - base
        rng = np.random.default_rng(seed + 1)
        bootstrap = np.array([rng.choice(difference, len(difference), replace=True).mean()
                              for _ in range(2000)])
        report[gain] = {"baseline_ndcg": float(base.mean()),
                        "selected_ndcg": float(selected.mean()),
                        "paired_delta": float(difference.mean()),
                        "bootstrap_95pct_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
                        "queries_improved": int((difference > 0).sum()),
                        "queries_worsened": int((difference < 0).sum()),
                        "queries_unchanged": int((difference == 0).sum())}
    if config_hash != digest(args.config_file):
        raise RuntimeError("Configuration changed during the audit")
    report["seconds"] = round(perf_counter() - started, 3)
    write_json(args.report_file, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
