"""Query-local transformations of the supplied numeric features only."""

import numpy as np
import pandas as pd


MODES = ("raw", "rank", "relative")


def transform(X, query_ids, mode="raw"):
    """Keep originals and optionally append within-query ranks/normalization.

    No fitted state, labels, object identifiers, or other queries enter a row's
    transformed features. NaNs are retained for the tree learners. A one-item
    query and a constant feature both receive zero in derived columns.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown feature mode: {mode}")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or len(X) != len(query_ids) or np.isinf(X).any():
        raise ValueError("Invalid feature matrix or query identifiers")
    if mode == "raw":
        return X
    frame = pd.DataFrame(X)
    groups = frame.groupby(np.asarray(query_ids), sort=False)
    low = groups.transform("min")
    high = groups.transform("max")
    constant = high.eq(low)
    ranks = groups.rank(method="average", na_option="keep")
    count = groups.transform("count")
    percentile = ((ranks - 1) / (count - 1).replace(0, 1)).mask(constant & frame.notna(), 0)
    arrays = [X, percentile.to_numpy(dtype=np.float32)]
    if mode == "relative":
        minimum_maximum = ((frame - low) / (high - low).replace(0, 1))
        arrays.append(minimum_maximum.to_numpy(dtype=np.float32))
    result = np.concatenate(arrays, axis=1)
    if np.isinf(result).any():
        raise ValueError("Feature transformation overflowed")
    return result


def normalize_scores(scores, query_ids, mode="zscore"):
    """Make component predictions comparable before a weighted ensemble."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(query_ids),) or not np.isfinite(scores).all():
        raise ValueError("Invalid predictions")
    if mode == "none":
        return scores
    series = pd.Series(scores)
    groups = series.groupby(np.asarray(query_ids), sort=False)
    if mode == "zscore":
        std = groups.transform("std", ddof=0)
        out = (series - groups.transform("mean")) / std.mask(std < 1e-12, 1)
    elif mode == "rank":
        n = groups.transform("count")
        out = (groups.rank(method="average") - 1) / (n - 1).clip(lower=1)
    else:
        raise ValueError(f"Unknown score normalization: {mode}")
    return out.to_numpy(dtype=np.float64)
