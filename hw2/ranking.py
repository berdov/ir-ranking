"""Hand-written lexical ranking formulas and query-level parameter selection.

Only the supplied training labels select coefficients. Test labels are never
read by a scorer; normalizing a blend uses candidate scores within each query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np


@dataclass
class RankingData:
    query_ids: list[str]
    doc_ids: list[str]
    labels: np.ndarray
    is_train: np.ndarray
    term_pair: np.ndarray
    qtf: np.ndarray
    tf_title: np.ndarray
    tf_body: np.ndarray
    tf_url: np.ndarray
    idf: np.ndarray
    idf_url: np.ndarray
    collection_prob: np.ndarray
    length_title: np.ndarray
    length_body: np.ndarray
    length_url: np.ndarray
    avg_title: float
    avg_body: float
    avg_url: float
    tfidf: np.ndarray
    title_coverage: np.ndarray
    body_coverage: np.ndarray
    phrase_title: np.ndarray
    phrase_body: np.ndarray
    proximity: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def _query_codes(self) -> np.ndarray:
        """Query IDs must not be mutated after constructing the data object."""
        return np.unique(np.asarray(self.query_ids), return_inverse=True)[1]


def _positive(config: dict, name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _b(config: dict, name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return value


def _weight(config: dict, name: str, default: float) -> float:
    value = float(config.get(name, default))
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _length_norm(lengths: np.ndarray, average: float, b: float) -> np.ndarray:
    # A completely empty field contributes no tf. Its undefined average does
    # not need an artificial epsilon that would change nonempty documents.
    relative = lengths / average if average > 0 else np.zeros_like(lengths)
    return 1.0 - b + b * relative


def _sum_terms(data: RankingData, values: np.ndarray) -> np.ndarray:
    return np.bincount(data.term_pair, weights=values, minlength=len(data.query_ids))


def _minmax(data: RankingData, values: np.ndarray) -> np.ndarray:
    """Label-free minmax, independently in every supplied candidate set."""
    if not len(values):
        return values.copy()
    codes = data._query_codes
    n_queries = int(codes.max()) + 1
    low = np.full(n_queries, np.inf)
    high = np.full(n_queries, -np.inf)
    np.minimum.at(low, codes, values)
    np.maximum.at(high, codes, values)
    width = high[codes] - low[codes]
    return np.divide(values - low[codes], width, out=np.zeros_like(values), where=width > 0)


def score(data: RankingData, config: dict[str, Any]) -> np.ndarray:
    """Return one finite, larger-is-better score per candidate pair.

    Configs are ordinary JSON dictionaries with a ``model`` key. BM25 and
    BM25F use each distinct query term once, following the seminar; Dirichlet
    uses qtf, as required by query likelihood. No labels are accessed here.
    """
    model = config.get("model", "bm25f")
    pair = data.term_pair
    if model == "tfidf":
        result = np.asarray(data.tfidf, dtype=np.float64).copy()
    elif model == "bm25":
        k1 = _positive(config, "k1", 1.2)
        b = _b(config, "b", 0.6)
        tf = data.tf_title + data.tf_body
        lengths = data.length_title + data.length_body
        norm = _length_norm(lengths, data.avg_title + data.avg_body, b)[pair]
        denominator = tf + k1 * norm
        saturated = np.divide(tf * (k1 + 1.0), denominator,
                              out=np.zeros_like(tf), where=denominator > 0)
        result = _sum_terms(data, data.idf * saturated)
    elif model == "bm25f":
        k1 = _positive(config, "k1", 1.2)
        title_weight = _weight(config, "w_title", 2.0)
        url_weight = _weight(config, "w_url", 0.0)
        title_norm = _length_norm(data.length_title, data.avg_title,
                                  _b(config, "b_title", 0.3))[pair]
        body_norm = _length_norm(data.length_body, data.avg_body,
                                 _b(config, "b_body", 0.6))[pair]
        # Normalize each field first, add its weighted tf, then saturate ONCE.
        # In particular, a linear sum of two separately saturated BM25 scores
        # is not this BM25F formula.
        tf = title_weight * np.divide(data.tf_title, title_norm,
                                      out=np.zeros_like(data.tf_title), where=title_norm > 0)
        tf += np.divide(data.tf_body, body_norm,
                        out=np.zeros_like(data.tf_body), where=body_norm > 0)
        idf = data.idf
        if url_weight > 0:
            url_norm = _length_norm(data.length_url, data.avg_url,
                                    _b(config, "b_url", 0.3))[pair]
            tf += url_weight * np.divide(data.tf_url, url_norm,
                                        out=np.zeros_like(data.tf_url), where=url_norm > 0)
            idf = data.idf_url
        result = _sum_terms(data, idf * tf * (k1 + 1.0) / (tf + k1))
    elif model == "lm":
        mu = _positive(config, "mu", 1000.0)
        known = data.collection_prob > 0
        gain = np.zeros(len(pair), dtype=np.float64)
        tf = data.tf_title + data.tf_body
        gain[known] = data.qtf[known] * np.log1p(
            tf[known] / (mu * data.collection_prob[known]))
        # OOV (p_C=0) terms are excluded from both the sum and |q|. Including
        # them only in |q| would penalize length for words absent everywhere.
        query_length = _sum_terms(data, np.where(known, data.qtf, 0.0))
        lengths = data.length_title + data.length_body
        # Unlike a retrieval implementation, score ALL given candidates,
        # including zero-overlap documents, which retain this length penalty.
        result = _sum_terms(data, gain) - query_length * np.log1p(lengths / mu)
    elif model == "blend":
        base = config.get("base", {"model": "bm25f"})
        if base.get("model", "bm25f") == "blend":
            raise ValueError("Nested blend configs are not supported")
        lm_weight = _b(config, "lm_weight", 0.0)
        result = (1.0 - lm_weight) * _minmax(data, score(data, base))
        if lm_weight:
            result += lm_weight * _minmax(data, score(data, {
                "model": "lm", "mu": _positive(config, "lm_mu", 1000.0)}))
        phrase_weight = _weight(config, "phrase_weight", 0.0)
        if phrase_weight:
            result += phrase_weight * _minmax(data, data.phrase_title + 0.5 * data.phrase_body)
        coverage_weight = _weight(config, "coverage_weight", 0.0)
        if coverage_weight:
            result += coverage_weight * _minmax(data, data.title_coverage)
        proximity_weight = _weight(config, "proximity_weight", 0.0)
        if proximity_weight:
            result += proximity_weight * _minmax(data, data.proximity)
    else:
        raise ValueError(f"Unknown ranking model: {model!r}")
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (len(data.query_ids),) or not np.all(np.isfinite(result)):
        raise ValueError("Scoring produced missing or non-finite candidate scores")
    return result


class _Metric:
    """Vectorized per-query evaluation with cached groups and ideal DCGs."""

    def __init__(self, labels, query_ids, doc_ids, k=10):
        if k < 1:
            raise ValueError("k must be positive")
        self.labels = np.asarray(labels, dtype=np.float64)
        if len(self.labels) != len(query_ids) or len(doc_ids) != len(query_ids):
            raise ValueError("Metric inputs must have equal lengths")
        if np.any(~np.isfinite(self.labels)) or np.any(self.labels < 0):
            raise ValueError("Metrics require finite nonnegative relevance labels")
        self.query_ids, self.codes = np.unique(np.asarray(query_ids, dtype=str), return_inverse=True)
        self.n_queries = len(self.query_ids)
        self.ties = np.asarray(doc_ids, dtype=str)
        self.k = k
        self.discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
        # Kaggle Evaluation uses 2**rel - 1. The seminar demonstrates linear
        # rel, so copying its dcg() verbatim would optimize another metric.
        self.gains = np.exp2(self.labels) - 1.0
        sizes = np.bincount(self.codes, minlength=self.n_queries)
        starts = np.r_[0, np.cumsum(sizes[:-1])] if len(sizes) else np.array([], dtype=int)
        self.positions = np.arange(len(self.labels)) - np.repeat(starts, sizes)
        self.top = self.positions < k
        self.ideal = self._dcg(np.lexsort((-self.gains, self.codes)))

    def _dcg(self, order):
        ranked = order[self.top]
        return np.bincount(self.codes[ranked],
                           weights=self.gains[ranked] * self.discounts[self.positions[self.top]],
                           minlength=self.n_queries)

    def per_query(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != self.labels.shape or np.any(~np.isfinite(scores)):
            raise ValueError("Metric scores must be finite and match the labels")
        # Training qrels have no ObjectId. Use lexicographic DocumentId as a
        # fixed, label-independent tie key. Kaggle breaks test ties using
        # ObjectId instead; the CSV preserves raw scores for that evaluator.
        dcg = self._dcg(np.lexsort((self.ties, -scores, self.codes)))
        return np.divide(dcg, self.ideal, out=np.zeros_like(dcg, dtype=np.float64), where=self.ideal > 0)


def per_query_ndcg(labels, scores, query_ids, doc_ids, k=10) -> dict[str, float]:
    metric = _Metric(labels, query_ids, doc_ids, k)
    return {str(q): float(v) for q, v in zip(metric.query_ids, metric.per_query(scores))}


def ndcg(labels, scores, query_ids, doc_ids, k=10) -> float:
    """Mean exponential-gain nDCG; a query with zero ideal gain scores zero."""
    values = _Metric(labels, query_ids, doc_ids, k).per_query(scores)
    return float(values.mean()) if len(values) else 0.0


def _training_subset(data: RankingData) -> RankingData:
    """Exclude test pairs before the tuning procedure sees relevance labels."""
    keep = np.asarray(data.is_train, dtype=bool)
    rows = np.flatnonzero(keep)
    mapping = np.full(len(keep), -1, dtype=np.int64)
    mapping[rows] = np.arange(len(rows))
    entries = keep[data.term_pair]
    pair_fields = ("labels", "length_title", "length_body", "length_url", "tfidf",
                   "title_coverage", "body_coverage", "phrase_title", "phrase_body", "proximity")
    term_fields = ("qtf", "tf_title", "tf_body", "tf_url", "idf", "idf_url", "collection_prob")
    args = {name: np.asarray(getattr(data, name))[rows] for name in pair_fields}
    args.update({name: np.asarray(getattr(data, name))[entries] for name in term_fields})
    return RankingData(query_ids=[data.query_ids[i] for i in rows],
                       doc_ids=[data.doc_ids[i] for i in rows],
                       is_train=np.ones(len(rows), dtype=bool),
                       term_pair=mapping[data.term_pair[entries]].astype(np.int32),
                       avg_title=data.avg_title, avg_body=data.avg_body,
                       avg_url=data.avg_url, metadata=data.metadata.copy(), **args)


def select_parameters(data: RankingData, seed: int = 42,
                      holdout_fraction: float = 0.3) -> tuple[dict, dict]:
    """Choose a small formula grid on tune queries; holdout is report-only.

    The selected config is fixed before any holdout score is inspected. No
    refit or extra model selection on holdout or Kaggle follows this function.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    train = _training_subset(data)
    if not len(train.query_ids):
        raise ValueError("Parameter selection requires labeled training queries")
    metric = _Metric(train.labels, train.query_ids, train.doc_ids)
    n_queries = metric.n_queries
    n_holdout = min(n_queries - 1, max(1, int(round(n_queries * holdout_fraction))))
    order = np.random.default_rng(seed).permutation(n_queries)
    holdout_mask = np.zeros(n_queries, dtype=bool)
    holdout_mask[order[:n_holdout]] = True
    tune_mask = ~holdout_mask
    tune_rows = tune_mask[metric.codes]
    tune_metric = _Metric(train.labels[tune_rows], np.asarray(train.query_ids)[tune_rows],
                          np.asarray(train.doc_ids)[tune_rows])
    trials: list[dict] = []
    family_best: dict[str, dict] = {}

    def evaluate(config):
        values = tune_metric.per_query(score(train, config)[tune_rows])
        trial = {"config": config.copy(), "tune_ndcg": float(values.mean())}
        trials.append(trial)
        family = config["model"]
        previous = family_best.get(family)
        # Keep the earlier, simpler configuration when scores tie exactly.
        if previous is None or trial["tune_ndcg"] > previous["tune_ndcg"] + 1e-12:
            family_best[family] = trial
        return trial

    evaluate({"model": "tfidf"})
    for k1 in (0.6, 1.2, 2.0):
        for b in (0.0, 0.3, 0.6, 0.9):
            evaluate({"model": "bm25", "k1": k1, "b": b})

    for k1 in (0.6, 1.2, 2.0):
        for b_body in (0.0, 0.3, 0.6, 0.9):
            for w_title in (1.0, 2.0, 4.0, 8.0):
                evaluate({"model": "bm25f", "k1": k1, "b_body": b_body,
                          "b_title": 0.3, "w_title": w_title, "w_url": 0.0})
    bm25f = family_best["bm25f"]["config"].copy()
    for b_title in (0.0, 0.3, 0.6):
        evaluate({**bm25f, "b_title": b_title})
    bm25f = family_best["bm25f"]["config"].copy()
    for w_url in (0.0, 0.25, 0.5):
        evaluate({**bm25f, "w_url": w_url, "b_url": 0.3})

    for mu in (200.0, 1000.0, 3000.0, 10000.0):
        evaluate({"model": "lm", "mu": mu})
    bm25f = family_best["bm25f"]["config"].copy()
    mu = family_best["lm"]["config"]["mu"]
    for lm_weight in (0.0, 0.1, 0.25, 0.5):
        evaluate({"model": "blend", "base": bm25f.copy(), "lm_mu": mu,
                  "lm_weight": lm_weight, "phrase_weight": 0.0,
                  "coverage_weight": 0.0, "proximity_weight": 0.0})
    # Only three small, predeclared coordinate steps. No holdout feedback.
    for key, grid in (("phrase_weight", (0.0, 0.05, 0.1, 0.2)),
                      ("coverage_weight", (0.0, 0.05, 0.1, 0.2)),
                      ("proximity_weight", (0.0, 0.02, 0.05))):
        base = family_best["blend"]["config"].copy()
        for value in grid:
            evaluate({**base, key: value})

    best = trials[0]
    for trial in trials[1:]:
        if trial["tune_ndcg"] > best["tune_ndcg"] + 1e-12:
            best = trial
    selected = best["config"].copy()
    # Holdout is first evaluated only after selected has been frozen above.
    per_family = {}
    values_by_family = {}
    for family, trial in family_best.items():
        values = metric.per_query(score(train, trial["config"]))
        values_by_family[family] = values
        per_family[family] = {"config": trial["config"],
                              "tune_ndcg": float(values[tune_mask].mean()),
                              "holdout_ndcg": float(values[holdout_mask].mean()) if n_holdout else None,
                              "train_ndcg": float(values.mean())}
    selected_values = metric.per_query(score(train, selected))
    differences = selected_values[holdout_mask] - values_by_family["tfidf"][holdout_mask]
    if len(differences):
        bootstrap_rng = np.random.default_rng(seed + 1)
        samples = bootstrap_rng.choice(differences, size=(500, len(differences)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        comparison = {"mean_delta": float(differences.mean()),
                      "ci95": [float(low), float(high)], "resamples": 500,
                      "seed": seed + 1, "better_queries": int((differences > 0).sum()),
                      "worse_queries": int((differences < 0).sum()),
                      "tied_queries": int((differences == 0).sum())}
    else:
        comparison = {"mean_delta": None, "ci95": None, "resamples": 0,
                      "note": "Only one training query; no independent holdout is possible."}
    report = {
        "metric": "nDCG@10, gain=2**relevance-1, mean over queries, IDCG=0 gives 0",
        "train_ties": "lexicographic DocumentId; training qrels have no ObjectId",
        "test_ties": "Kaggle uses ascending ObjectId for equal Score; CSV scores are unmodified",
        "selection_policy": "Only tune queries select all parameters; holdout is report-only",
        "seed": seed, "holdout_fraction": holdout_fraction,
        "train_queries": n_queries, "tune_queries": int(tune_mask.sum()),
        "holdout_queries": n_holdout, "train_pairs": len(train.query_ids),
        "tune_pairs": int(tune_rows.sum()), "holdout_pairs": int((~tune_rows).sum()),
        "splits": {"tune_query_ids": metric.query_ids[tune_mask].tolist(),
                   "holdout_query_ids": metric.query_ids[holdout_mask].tolist()},
        "selected": selected, "per_family": per_family,
        "selected_metrics": {"tune_ndcg": float(selected_values[tune_mask].mean()),
                             "holdout_ndcg": float(selected_values[holdout_mask].mean()) if n_holdout else None,
                             "train_ndcg": float(selected_values.mean())},
        "selected_vs_baseline_holdout_bootstrap": comparison,
        "evaluated_configurations": len(trials), "candidates": trials,
    }
    return selected, report
