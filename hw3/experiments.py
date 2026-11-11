#!/usr/bin/env python3
"""Compare predefined rankers using train.csv only and disjoint query splits.

The final configuration is selected on tuning queries and written BEFORE the
untouched holdout is scored. The standard solution.py --train does not run this
search: it simply refits the selected configuration on all training queries.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from data import load_train
from features import normalize_scores
from metrics import mean_ndcg, per_query_ndcg
from modeling import fit_component, predict_component, save_models, subset
from solution import digest


def candidates(gain="exp2", quick=False):
    cb_metric = "NDCG:top=10;type=" + ("Exp" if gain == "exp2" else "Base") + ";denominator=LogPosition"
    output = [{"name": "baseline_yetirank200", "family": "catboost", "features": "raw",
               "iterations": 200, "gain": gain, "params": {}, "seed": 0}]
    for mode in ("raw", "rank", "relative"):
        output.append({"name": f"lgb_{mode}_31", "family": "lightgbm", "features": mode,
                       "iterations": 1500, "gain": gain, "params": {"num_leaves": 31}})
    for mode in ("raw", "relative"):
        output.append({"name": f"lgb_{mode}_63", "family": "lightgbm", "features": mode,
                       "iterations": 1500, "gain": gain,
                       "params": {"num_leaves": 63, "min_data_in_leaf": 80, "lambda_l2": 10.}})
    output.append({"name": "lgb_xendcg_relative", "family": "lightgbm", "features": "relative",
                   "iterations": 1500, "gain": gain,
                   "params": {"objective": "rank_xendcg", "num_leaves": 31}})
    for mode in ("raw", "rank"):
        output.append({"name": f"cat_yeti_{mode}", "family": "catboost", "features": mode,
                       "iterations": 1200, "gain": gain,
                       "params": {"depth": 6, "learning_rate": .04, "l2_leaf_reg": 5.,
                                  "eval_metric": cb_metric}})
    output.append({"name": "cat_ndcg_rank", "family": "catboost", "features": "rank",
                   "iterations": 1200, "gain": gain,
                   "params": {"depth": 6, "learning_rate": .04, "l2_leaf_reg": 5.,
                              "eval_metric": cb_metric,
                              "loss_function": "YetiRank:mode=NDCG;top=10;dcg_type="
                              + ("Exp" if gain == "exp2" else "Base")
                              + ";dcg_denominator=LogPosition"}})
    if quick:
        output = output[:3]
        for item in output:
            item["iterations"] = 8
    return output


def make_split(query_ids, seed=42):
    unique = np.random.default_rng(seed).permutation(np.unique(query_ids))
    if len(unique) < 7:
        raise ValueError("At least 7 queries are needed for three disjoint subsets")
    n_eval = max(1, len(unique) // 7)
    return {"fit": unique[2 * n_eval:], "tuning": unique[n_eval:2 * n_eval],
            "holdout": unique[:n_eval]}


def score(dataset, scores, gain):
    return mean_ndcg(dataset.labels, scores, dataset.query_ids, gain=gain)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def choose(cases, predictions, tune, seed=42, gain="exp2"):
    """Only tuning labels are accessible here; holdout/test cannot affect choice."""
    def config(parts, normalization="none"):
        return {"seed": seed, "score_normalization": normalization, "components": parts}
    choices = []
    for name, case in cases.items():
        component = dict(case, weight=1.)
        choices.append((score(tune, predictions[name], gain), name, config([component])))
    best_by_family = {}
    for value, name, proposal in choices:
        family = proposal["components"][0]["family"]
        if family not in best_by_family or value > best_by_family[family][0]:
            best_by_family[family] = (value, name, proposal["components"][0])
    if len(best_by_family) == 2:
        (_, name_a, a), (_, name_b, b) = best_by_family.values()
        for normalization in ("zscore", "rank"):
            A = normalize_scores(predictions[name_a], tune.query_ids, normalization)
            B = normalize_scores(predictions[name_b], tune.query_ids, normalization)
            for weight in (.25, .5, .75):
                prediction = weight * A + (1 - weight) * B
                proposal = config([dict(a, weight=weight), dict(b, weight=1 - weight)], normalization)
                name = f"{name_a}+{name_b}:{normalization}:{weight}"
                choices.append((score(tune, prediction, gain), name, proposal))
    # Stable ordering favors a single simpler candidate in the event of an exact tie.
    best = max(choices, key=lambda item: item[0])
    return best, [{"name": name, "tuning_ndcg": value} for value, name, _ in choices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gain", choices=("exp2", "linear"), default="exp2")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Only a synthetic smoke check; not a valid baseline run")
    args = parser.parse_args()
    started = perf_counter()
    data = load_train(args.data_dir / "train.csv")
    split = make_split(data.query_ids, args.seed)
    indices = {name: np.flatnonzero(np.isin(data.query_ids, ids)) for name, ids in split.items()}
    fit, tune = subset(data, indices["fit"]), subset(data, indices["tuning"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    report = {"train_sha256": digest(args.data_dir / "train.csv"), "seed": args.seed,
              "gain": args.gain, "k": 10, "zero_idcg": 0, "ties": "original train row order",
              "quick_smoke_only": args.quick,
              "split_queries": {key: value.tolist() for key, value in split.items()},
              "split_rows": {key: len(value) for key, value in indices.items()}, "candidates": []}
    write_json(args.output_dir / "splits.json", report)
    fitted, predictions = {}, {}
    for candidate in candidates(args.gain, args.quick):
        begin = perf_counter()
        name = candidate["name"]
        baseline = name == "baseline_yetirank200"
        model, config = fit_component(fit, candidate, None if baseline else tune,
                                       seed=candidate.get("seed", args.seed), threads=args.threads)
        prediction = predict_component(model, tune, config, args.threads)
        fitted[name], predictions[name] = config, prediction
        result = {"name": name, "config": config, "seconds": round(perf_counter() - begin, 3),
                  "tuning_ndcg": score(tune, prediction, args.gain),
                  "tuning_ndcg_other_gain": score(tune, prediction, "linear" if args.gain == "exp2" else "exp2")}
        report["candidates"].append(result)
        save_models(model_dir / f"{name}.bin", [model],
                    {"seed": candidate.get("seed", args.seed), "score_normalization": "none", "components": [config]},
                    data.feature_names, {"purpose": "tuning", "fit_queries": len(split["fit"])})
        np.savez_compressed(model_dir / f"{name}.npz", predictions=prediction)
        write_json(args.output_dir / "experiments.json", report)
        print(json.dumps(result), flush=True)
        del model
    (best_score, name, best_config), choices = choose(fitted, predictions, tune, args.seed, args.gain)
    # Persist the decision before accessing holdout labels. Never change it based
    # on the subsequent audit; this prevents hidden tuning on the final holdout.
    best_config["status"] = "smoke_test_only" if args.quick else "selected_on_training_queries"
    best_config["selection"] = {"tuning_ndcg": best_score, "name": name, "gain": args.gain}
    write_json(args.output_dir / "best_config.json", best_config)
    report["selection"] = {"name": name, "tuning_ndcg": best_score, "choices": choices}
    write_json(args.output_dir / "experiments.json", report)

    holdout = subset(data, indices["holdout"])
    # Retain original CSV order when merging the two training subsets.
    audit_train = subset(data, np.sort(np.concatenate([indices["fit"], indices["tuning"]])))
    baseline_config = fitted["baseline_yetirank200"]
    baseline_model, _ = fit_component(audit_train, baseline_config, seed=0, threads=args.threads)
    baseline_predictions = predict_component(baseline_model, holdout, baseline_config, args.threads)
    del baseline_model
    final_predictions = np.zeros(len(holdout.query_ids), dtype=np.float64)
    weight_sum = 0.
    for component in best_config["components"]:
        model, _ = fit_component(audit_train, component, seed=component.get("seed", args.seed), threads=args.threads)
        prediction = predict_component(model, holdout, component, args.threads)
        weight = component.get("weight", 1.)
        final_predictions += weight * normalize_scores(prediction, holdout.query_ids, best_config["score_normalization"])
        weight_sum += weight
        del model
    final_predictions /= weight_sum
    _, baseline_per_query = per_query_ndcg(holdout.labels, baseline_predictions, holdout.query_ids, gain=args.gain)
    _, selected_per_query = per_query_ndcg(holdout.labels, final_predictions, holdout.query_ids, gain=args.gain)
    delta = selected_per_query - baseline_per_query
    rng = np.random.default_rng(args.seed + 1)
    boot = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(2000)])
    report["holdout"] = {"baseline_ndcg": float(baseline_per_query.mean()),
                         "selected_ndcg": float(selected_per_query.mean()),
                         "paired_delta": float(delta.mean()),
                         "bootstrap_95pct_ci": np.quantile(boot, [.025, .975]).tolist(),
                         "queries": len(split["holdout"]), "used_for_selection": False,
                         "baseline_other_gain": score(holdout, baseline_predictions, "linear" if args.gain == "exp2" else "exp2"),
                         "selected_other_gain": score(holdout, final_predictions, "linear" if args.gain == "exp2" else "exp2")}
    report["seconds"] = round(perf_counter() - started, 3)
    write_json(args.output_dir / "experiments.json", report)
    print(json.dumps({"selected": report["selection"]["name"], "holdout": report["holdout"],
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
