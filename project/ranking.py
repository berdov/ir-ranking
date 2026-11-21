"""Query-level ranking metrics, transformations and portable tree rankers.

Only supplied numerical features are used. Query/document IDs define groups and
ties; they are never columns passed to a learner. Model files are native CBM or
LightGBM text, with a JSON manifest rather than executable pickle objects.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np


METRIC = {"name": "nDCG@10", "gain": "2**label - 1", "discount": "1/log2(rank+1)",
          "aggregation": "equal weight per query", "zero_idcg": 0,
          "training_ties": "lexicographic DocumentId"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def subset(bundle, rows):
    rows = np.asarray(rows)
    return SimpleNamespace(
        X=np.asarray(bundle.X)[rows], labels=np.asarray(bundle.labels)[rows],
        query_ids=np.asarray(bundle.query_ids).astype(str)[rows],
        doc_ids=np.asarray(bundle.doc_ids).astype(str)[rows],
        object_ids=np.asarray(bundle.object_ids).astype(str)[rows],
        is_train=np.asarray(bundle.is_train, dtype=bool)[rows],
        feature_names=list(bundle.feature_names), metadata=dict(bundle.metadata))


def validate_bundle(bundle):
    X = np.asarray(bundle.X)
    n = len(X)
    if X.ndim != 2 or X.shape[1] != len(bundle.feature_names):
        raise ValueError("Feature names do not match the two-dimensional matrix")
    if len(set(bundle.feature_names)) != len(bundle.feature_names):
        raise ValueError("Duplicate feature names")
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite")
    for key in ("labels", "query_ids", "doc_ids", "object_ids", "is_train"):
        if np.asarray(getattr(bundle, key)).shape != (n,):
            raise ValueError(f"Invalid {key} array")
    train = np.asarray(bundle.is_train, dtype=bool)
    qids = np.asarray(bundle.query_ids).astype(str)
    if set(qids[train]) & set(qids[~train]):
        raise ValueError("A query appears in both training and test rows")
    labels = np.asarray(bundle.labels)
    if np.any((labels[train] < 0) | (labels[train] > 3)) or np.any(labels[train] % 1):
        raise ValueError("Training labels must be integers from 0 to 3")
    if np.any(labels[~train] != -1):
        raise ValueError("Test rows must have the sentinel label -1")


class NDCG:
    """Reusable vectorized exponential-gain nDCG@10 evaluator."""

    def __init__(self, labels, query_ids, doc_ids=None, k=10):
        labels = np.asarray(labels, dtype=np.float64)
        query_ids = np.asarray(query_ids).astype(str)
        if labels.ndim != 1 or query_ids.shape != labels.shape or not len(labels):
            raise ValueError("Labels and groups must be nonempty, equal-length vectors")
        if not np.isfinite(labels).all() or np.any(labels < 0):
            raise ValueError("Metric labels must be finite and nonnegative")
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        self.query_ids, self.codes = np.unique(query_ids, return_inverse=True)
        self.ties = (np.arange(len(labels)) if doc_ids is None
                     else np.asarray(doc_ids).astype(str))
        if self.ties.shape != labels.shape:
            raise ValueError("Document tie keys must match the labels")
        self.gain = np.exp2(labels) - 1
        if not np.isfinite(self.gain).all():
            raise ValueError("Metric gains overflowed")
        sizes = np.bincount(self.codes)
        starts = np.r_[0, np.cumsum(sizes[:-1])]
        self.positions = np.arange(len(labels)) - np.repeat(starts, sizes)
        self.top = self.positions < k
        self.discounts = 1.0 / np.log2(np.arange(k) + 2)
        self.ideal = self._dcg(np.lexsort((-self.gain, self.codes)))

    def _dcg(self, order):
        selected = order[self.top]
        return np.bincount(self.codes[selected], weights=self.gain[selected]
                           * self.discounts[self.positions[self.top]],
                           minlength=len(self.query_ids))

    def per_query(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != self.gain.shape or not np.isfinite(scores).all():
            raise ValueError("Scores must be finite and match the metric labels")
        dcg = self._dcg(np.lexsort((self.ties, -scores, self.codes)))
        return np.divide(dcg, self.ideal, out=np.zeros_like(dcg), where=self.ideal > 0)

    def mean(self, scores):
        return float(self.per_query(scores).mean())


def transform(X, query_ids, mode="raw"):
    """Append percentile ranks inside each complete query candidate set.

    Equal values receive average ranks; constant columns and singleton queries
    receive zero derived values. No state is fitted, and labels are not read.
    """
    X = np.asarray(X, dtype=np.float32)
    qids = np.asarray(query_ids).astype(str)
    if X.ndim != 2 or len(X) != len(qids) or not np.isfinite(X).all():
        raise ValueError("Invalid numeric feature matrix")
    if mode == "raw":
        return X
    if mode != "rank":
        raise ValueError(f"Unknown feature transformation: {mode}")
    import pandas as pd
    frame = pd.DataFrame(X)
    groups = frame.groupby(qids, sort=False)
    ranks = groups.rank(method="average")
    sizes = groups.transform("count")
    percentile = (ranks - 1) / (sizes - 1).clip(lower=1)
    constant = groups.transform("max").eq(groups.transform("min"))
    percentile = percentile.mask(constant, 0)
    return np.concatenate((X, percentile.to_numpy(dtype=np.float32)), axis=1)


def normalize_scores(scores, query_ids, mode="none"):
    scores = np.asarray(scores, dtype=np.float64)
    qids = np.asarray(query_ids).astype(str)
    if scores.shape != qids.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid score vector")
    if mode == "none":
        return scores.copy()
    if mode != "zscore":
        raise ValueError(f"Unknown score normalization: {mode}")
    _, codes = np.unique(qids, return_inverse=True)
    size = np.bincount(codes)
    mean = np.bincount(codes, weights=scores) / size
    centered = scores - mean[codes]
    std = np.sqrt(np.bincount(codes, weights=centered ** 2) / size)
    return centered / np.where(std[codes] > 1e-12, std[codes], 1)


def _columns(bundle, names):
    lookup = {name: index for index, name in enumerate(bundle.feature_names)}
    if not names or len(set(names)) != len(names) or any(name not in lookup for name in names):
        raise ValueError("Model requires a missing or duplicated feature")
    return np.asarray(bundle.X)[:, [lookup[name] for name in names]]


def fit_component(train, config, validation=None, threads=4):
    if threads < 1 or np.any(np.asarray(train.labels) < 0):
        raise ValueError("Training needs labelled rows and a positive thread count")
    if validation is not None and set(train.query_ids) & set(validation.query_ids):
        raise ValueError("Training and validation queries overlap")
    if config["family"] == "column":
        _columns(train, [config["column"]])
        return None, dict(config)
    order = np.argsort(np.asarray(train.query_ids).astype(str), kind="stable")
    qids = np.asarray(train.query_ids).astype(str)[order]
    X = transform(_columns(train, config["columns"])[order], qids, config["feature_mode"])
    labels = np.asarray(train.labels)[order]
    V = vlabels = vqids = None
    if validation is not None:
        vorder = np.argsort(np.asarray(validation.query_ids).astype(str), kind="stable")
        vqids = np.asarray(validation.query_ids).astype(str)[vorder]
        V = transform(_columns(validation, config["columns"])[vorder], vqids, config["feature_mode"])
        vlabels = np.asarray(validation.labels)[vorder]
    rounds, seed = int(config["iterations"]), int(config.get("seed", 42))
    if rounds < 1:
        raise ValueError("Number of iterations must be positive")
    if config["family"] == "catboost":
        from catboost import CatBoostRanker, Pool
        params = {"loss_function": "YetiRank", "learning_rate": .04, "depth": 6,
                  "l2_leaf_reg": 5., "eval_metric": "NDCG:top=10;type=Exp;denominator=LogPosition"}
        params.update(config.get("params", {}))
        params.update(iterations=rounds, random_seed=seed, thread_count=threads,
                      task_type="CPU", allow_writing_files=False, verbose=False)
        model = CatBoostRanker(**params)
        kwargs = {}
        if validation is not None:
            kwargs = {"eval_set": Pool(V, label=vlabels, group_id=vqids),
                      "early_stopping_rounds": 100, "use_best_model": True}
        model.fit(Pool(X, label=labels, group_id=qids), **kwargs)
        rounds = model.tree_count_
    elif config["family"] == "lightgbm":
        import lightgbm as lgb
        params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10],
                  "label_gain": [0, 1, 3, 7], "lambdarank_truncation_level": 13,
                  "learning_rate": .04, "num_leaves": 31, "min_data_in_leaf": 80,
                  "lambda_l2": 5., "feature_fraction": .9, "verbosity": -1}
        params.update(config.get("params", {}))
        params.update(num_threads=threads, seed=seed, data_random_seed=seed,
                      feature_fraction_seed=seed, bagging_seed=seed,
                      deterministic=True, force_col_wise=True)
        pool = lgb.Dataset(X, label=labels, group=np.unique(qids, return_counts=True)[1])
        valid_sets, callbacks = None, []
        if validation is not None:
            valid_sets = [lgb.Dataset(V, label=vlabels,
                          group=np.unique(vqids, return_counts=True)[1], reference=pool)]
            callbacks = [lgb.early_stopping(100, first_metric_only=True, verbose=False)]
        model = lgb.train(params, pool, num_boost_round=rounds,
                          valid_sets=valid_sets, callbacks=callbacks)
        rounds = model.current_iteration()
    else:
        raise ValueError(f"Unknown model family: {config['family']}")
    return model, dict(config, iterations=int(rounds))


def predict_component(model, bundle, config, threads=4):
    if config["family"] == "column":
        scores = _columns(bundle, [config["column"]])[:, 0]
    else:
        X = transform(_columns(bundle, config["columns"]), bundle.query_ids, config["feature_mode"])
        if config["family"] == "catboost":
            scores = model.predict(X, thread_count=threads)
        elif config["family"] == "lightgbm":
            scores = model.predict(X, num_threads=threads)
        else:
            raise ValueError("Unknown model family")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(bundle.query_ids),) or not np.isfinite(scores).all():
        raise ValueError("Model emitted invalid predictions")
    return scores


def save_component(model, config, directory, stem):
    if config["family"] == "column":
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".cbm" if config["family"] == "catboost" else ".txt"
    filename = stem + suffix
    path = directory / filename
    temporary = directory / ("." + filename + ".tmp")
    model.save_model(str(temporary))
    os.replace(temporary, path)
    return filename


def load_component(directory, config, filename):
    if config["family"] == "column":
        return None
    if not filename or Path(filename).name != filename:
        raise ValueError("Invalid model file name")
    path = Path(directory) / filename
    if config["family"] == "catboost":
        from catboost import CatBoostRanker
        return CatBoostRanker().load_model(str(path))
    if config["family"] == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    raise ValueError("Unsupported model family")


def environment_versions():
    versions = {}
    for name in ("numpy", "pandas", "catboost", "lightgbm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions
