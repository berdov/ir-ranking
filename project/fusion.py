"""Train-query selection of neural reranking depth, weight and query budget.

The confidence gate is label-free at inference. Documents outside a reranked
prefix retain their lexical order and cannot jump ahead of that prefix.
"""
from __future__ import annotations

import numpy as np

from ranking import NDCG, json_hash


def _groups(query_ids):
    query_ids = np.asarray(query_ids).astype(str)
    order = np.argsort(query_ids, kind="stable")
    starts = np.r_[0, np.flatnonzero(query_ids[order][1:] != query_ids[order][:-1]) + 1]
    return np.split(order, starts[1:]) if len(order) else []


def _standardize(x):
    x = np.asarray(x, dtype=np.float64)
    scale = x.std()
    return (x - x.mean()) / scale if scale > 1e-12 else np.zeros_like(x)


def ambiguity(scores):
    """Normalized softmax entropy of within-query standardized lexical scores."""
    if len(scores) < 2:
        return 0.0
    z = _standardize(scores)
    weights = np.exp(z - z.max())
    probability = weights / weights.sum()
    return float(-np.dot(probability, np.log(np.maximum(probability, 1e-300))) / np.log(len(scores)))


def _validate(query_ids, doc_ids, lexical, config):
    qids = np.asarray(query_ids).astype(str)
    dids = np.asarray(doc_ids).astype(str)
    scores = np.asarray(lexical, dtype=np.float64)
    if qids.ndim != 1 or dids.shape != qids.shape or scores.shape != qids.shape:
        raise ValueError("Fusion requires equal-length ID and score vectors")
    if not np.isfinite(scores).all():
        raise ValueError("Lexical scores must be finite")
    weight = float(config.get("neural_weight", 0))
    top_k = int(config.get("top_k", 0))
    threshold = float(config.get("entropy_threshold", 0))
    if not 0 <= weight <= 1 or top_k < 0 or not 0 <= threshold <= 1.00000001:
        raise ValueError("Invalid neural weight, prefix depth or entropy threshold")
    return qids, dids, scores


def required_rows(query_ids, doc_ids, lexical, config):
    """Return neural inference mask without reading any relevance labels."""
    qids, dids, lexical = _validate(query_ids, doc_ids, lexical, config)
    needed = np.zeros(len(qids), dtype=bool)
    if config.get("neural_weight", 0) == 0:
        return needed
    for rows in _groups(qids):
        if ambiguity(lexical[rows]) + 1e-12 < config.get("entropy_threshold", 0):
            continue
        ranked = rows[np.lexsort((dids[rows], -lexical[rows]))]
        k = int(config.get("top_k", 0)) or len(rows)
        needed[ranked[:k]] = True
    return needed


def fuse_scores(query_ids, doc_ids, lexical, neural, available, config):
    qids, dids, lexical = _validate(query_ids, doc_ids, lexical, config)
    if float(config.get("neural_weight", 0)) == 0:
        return lexical.copy()
    neural = np.asarray(neural, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    if neural.shape != lexical.shape or available.shape != lexical.shape:
        raise ValueError("Neural score/mask rows do not match lexical rows")
    needed = required_rows(qids, dids, lexical, config)
    if np.any(needed & ~available) or not np.isfinite(neural[needed]).all():
        raise ValueError("Missing or nonfinite required neural scores")
    result = np.empty(len(lexical), dtype=np.float64)
    alpha = float(config["neural_weight"])
    for rows in _groups(qids):
        lexical_order = rows[np.lexsort((dids[rows], -lexical[rows]))]
        if not needed[rows].any():
            order = lexical_order
        else:
            k = int(config.get("top_k", 0)) or len(rows)
            prefix = lexical_order[:k]
            mixed = (1-alpha) * _standardize(lexical[prefix]) + alpha * _standardize(neural[prefix])
            reordered = prefix[np.lexsort((dids[prefix], -mixed))]
            order = np.r_[reordered, lexical_order[k:]]
        # Unique integer scores encode only the within-query ordering.
        result[order] = np.arange(len(rows), 0, -1, dtype=np.float64)
    return result


def select_fusion(labels, query_ids, doc_ids, lexical, neural, available):
    """A fixed, bounded grid evaluated exclusively on training tune queries."""
    qids = np.asarray(query_ids).astype(str)
    if any(not q.startswith("train:") for q in qids):
        raise ValueError("Fusion tuning accepts only labelled training queries")
    metric = NDCG(labels, qids, doc_ids)
    groups = _groups(qids)
    entropy = np.asarray([ambiguity(np.asarray(lexical)[rows]) for rows in groups])
    baseline = {"neural_weight": 0.0, "top_k": 0, "entropy_threshold": 0.0}
    trials = [{"config": baseline, "ndcg10": metric.mean(lexical),
               "neural_pairs": 0, "neural_queries": 0}]
    # One fixed grid; no access to the final holdout or Kaggle scores.
    for top_k in (10, 20, 0):
        for fraction in (.25, .5, 1.0):
            threshold = float(np.quantile(entropy, 1-fraction)) if fraction < 1 else 0.0
            for alpha in (.25, .5, .75, 1.0):
                config = {"neural_weight": alpha, "top_k": top_k,
                          "entropy_threshold": threshold,
                          "target_query_fraction_on_tune": fraction}
                needed = required_rows(qids, doc_ids, lexical, config)
                if np.any(needed & ~np.asarray(available, dtype=bool)):
                    continue
                scores = fuse_scores(qids, doc_ids, lexical, neural, available, config)
                trials.append({"config": config, "ndcg10": metric.mean(scores),
                               "neural_pairs": int(needed.sum()),
                               "neural_queries": len(np.unique(qids[needed]))})
    winner = max(trials, key=lambda trial: (trial["ndcg10"], -trial["neural_pairs"]))
    return {"config": winner["config"], "tune_ndcg10": winner["ndcg10"],
            "baseline_ndcg10": trials[0]["ndcg10"], "trials": trials,
            "tune_queries": len(groups), "tune_pairs": len(qids),
            "query_ids_sha256": json_hash(sorted(np.unique(qids).tolist())),
            "gate": "normalized entropy of softmax(zscore(lexical)), high entropy reranked",
            "selection_rule": "maximum tune nDCG@10, exact ties prefer fewer neural pairs"}
