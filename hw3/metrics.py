"""Explicit nDCG@k conventions for query-level ranking experiments.

The default gain is 2**label - 1, and queries with zero ideal DCG contribute 0.
Ties are deterministic: original row order unless explicit document/object keys
are supplied. All-integer keys sort numerically (arbitrary precision), other
keys lexicographically. Equal explicit keys retain the original row order.
"""

from __future__ import annotations

import re
import numpy as np


def _tie_ranks(keys: np.ndarray | None, size: int) -> np.ndarray:
    if keys is None:
        return np.arange(size, dtype=np.int64)
    keys = np.asarray(keys)
    if keys.ndim != 1 or len(keys) != size:
        raise ValueError("tie_breaker must have one key per row")
    text = keys.astype(str)
    if all(re.fullmatch(r"[+-]?\d+", value) for value in text):
        order = sorted(range(size), key=lambda index: int(text[index]))
    else:
        order = np.argsort(text, kind="stable")
    ranks = np.empty(size, dtype=np.int64)
    ranks[order] = np.arange(size, dtype=np.int64)
    return ranks


def per_query_ndcg(labels: np.ndarray, scores: np.ndarray, query_ids: np.ndarray,
                   k: int = 10, gain: str = "exp2",
                   tie_breaker: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return lexicographically sorted query IDs and their nDCG values.

    Only query IDs define groups, and every query has equal weight regardless of
    how many candidates it contains. Labels/scores must be finite; labels must
    be nonnegative. Integer labels are validated by the competition loader.
    """
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    query_ids = np.asarray(query_ids).astype(str)
    if labels.ndim != 1 or scores.ndim != 1 or query_ids.ndim != 1:
        raise ValueError("labels, scores and query_ids must be one-dimensional")
    if len(labels) != len(scores) or len(labels) != len(query_ids):
        raise ValueError("labels, scores and query_ids must have the same length")
    if not len(labels):
        raise ValueError("cannot evaluate an empty dataset")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("labels must be finite and nonnegative")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if gain == "exp2":
        with np.errstate(over="ignore"):
            gains = np.exp2(labels) - 1
        if not np.isfinite(gains).all():
            raise ValueError("labels are too large for exponential gain")
    elif gain == "linear":
        gains = labels
    else:
        raise ValueError("gain must be 'exp2' or 'linear'")

    tie_ranks = _tie_ranks(tie_breaker, len(labels))
    order = np.argsort(query_ids, kind="stable")
    ordered_qids = query_ids[order]
    starts = np.r_[0, np.flatnonzero(ordered_qids[1:] != ordered_qids[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    results = np.zeros(len(starts), dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, min(k, len(labels)) + 2))
    for index, (start, end) in enumerate(zip(starts, ends)):
        rows = order[start:end]
        count = min(k, len(rows))
        query_gains = gains[rows]
        ideal = np.sort(query_gains)[::-1][:count]
        idcg = float(np.dot(ideal, discounts[:count]))
        if idcg == 0:
            continue
        ranked = np.lexsort((tie_ranks[rows], -scores[rows]))[:count]
        dcg = float(np.dot(query_gains[ranked], discounts[:count]))
        results[index] = dcg / idcg
    return ordered_qids[starts], results


def mean_ndcg(labels: np.ndarray, scores: np.ndarray, query_ids: np.ndarray,
              k: int = 10, gain: str = "exp2",
              tie_breaker: np.ndarray | None = None) -> float:
    """Unweighted arithmetic mean of per-query nDCG."""
    _, values = per_query_ndcg(labels, scores, query_ids, k, gain, tie_breaker)
    return float(values.mean())
