"""CPU ranking models and a portable archive of native model files."""

import importlib.metadata
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from data import Dataset, group_sizes, grouped
from features import normalize_scores, transform

FORMAT_VERSION = 1


def subset(dataset, indices):
    return Dataset(dataset.X[indices], dataset.query_ids[indices],
                   None if dataset.labels is None else dataset.labels[indices],
                   None if dataset.object_ids is None else dataset.object_ids[indices],
                   dataset.feature_names)


def fit_component(dataset, config, validation=None, seed=42, threads=4):
    """Train with complete query groups; optional validation never joins train."""
    train, _ = grouped(dataset)
    valid, _ = grouped(validation) if validation is not None else (None, None)
    X = transform(train.X, train.query_ids, config.get("features", "raw"))
    V = (transform(valid.X, valid.query_ids, config.get("features", "raw"))
         if valid is not None else None)
    params = dict(config.get("params", {}))
    rounds = int(config["iterations"])
    if rounds < 1 or threads < 1 or dataset.labels is None:
        raise ValueError("Positive iterations/threads and training labels are required")
    gain = config.get("gain", "exp2")
    if gain not in {"exp2", "linear"}:
        raise ValueError("gain must be exp2 or linear")
    if config["family"] == "lightgbm":
        import lightgbm as lgb
        defaults = {
            "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10],
            "label_gain": [0, 1, 3, 7] if gain == "exp2" else [0, 1, 2, 3],
            "lambdarank_truncation_level": 13, "learning_rate": .04,
            "num_leaves": 31, "min_data_in_leaf": 40, "lambda_l2": 5.,
            "feature_fraction": .9, "verbosity": -1,
        }
        defaults.update(params)
        defaults.update(num_threads=threads, seed=seed, data_random_seed=seed,
                        feature_fraction_seed=seed, bagging_seed=seed,
                        deterministic=True, force_col_wise=True)
        pool = lgb.Dataset(X, label=train.labels, group=group_sizes(train.query_ids))
        valid_sets, callbacks = None, []
        if valid is not None:
            valid_sets = [lgb.Dataset(V, label=valid.labels,
                                     group=group_sizes(valid.query_ids), reference=pool)]
            callbacks.append(lgb.early_stopping(100, first_metric_only=True, verbose=False))
        model = lgb.train(defaults, pool, num_boost_round=rounds,
                          valid_sets=valid_sets, callbacks=callbacks)
        actual_rounds = model.current_iteration()
    elif config["family"] == "catboost":
        from catboost import CatBoostRanker, Pool
        defaults = {"loss_function": "YetiRank", "iterations": rounds,
                    "random_seed": seed, "thread_count": threads,
                    "allow_writing_files": False, "verbose": False, "task_type": "CPU"}
        defaults.update(params)
        # These settings are deliberately not configurable: the grading uses CPU.
        defaults.update(iterations=rounds, random_seed=seed, thread_count=threads,
                        allow_writing_files=False, verbose=False, task_type="CPU")
        model = CatBoostRanker(**defaults)
        pool = Pool(X, label=train.labels, group_id=train.query_ids)
        kwargs = {}
        if valid is not None:
            kwargs = {"eval_set": Pool(V, label=valid.labels, group_id=valid.query_ids),
                      "early_stopping_rounds": 100, "use_best_model": True}
        model.fit(pool, **kwargs)
        actual_rounds = model.tree_count_
    else:
        raise ValueError(f"Unknown model family: {config['family']}")
    fitted_config = dict(config, iterations=actual_rounds)
    return model, fitted_config


def predict_component(model, dataset, config, threads=4):
    X = transform(dataset.X, dataset.query_ids, config.get("features", "raw"))
    if config["family"] == "catboost":
        scores = model.predict(X, thread_count=threads)
    else:
        scores = model.predict(X, num_threads=threads)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(dataset.query_ids),) or not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite or incorrectly shaped predictions")
    return scores


def ensemble_predict(models, dataset, config, threads=4):
    components = config["components"]
    if len(models) != len(components) or not models:
        raise ValueError("Model components do not match the configuration")
    total = np.zeros(len(dataset.query_ids), dtype=np.float64)
    denominator = 0.
    for model, component in zip(models, components):
        weight = float(component.get("weight", 1.))
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("Ensemble weights must be finite and nonnegative")
        prediction = predict_component(model, dataset, component, threads)
        total += weight * normalize_scores(prediction, dataset.query_ids,
                                            config.get("score_normalization", "none"))
        denominator += weight
    if denominator <= 0:
        raise ValueError("At least one ensemble weight must be positive")
    return total / denominator


def save_models(path, models, config, feature_names, training_info):
    """Save a zip of JSON and native CBM/LightGBM files, without pickle."""
    metadata = {"format_version": FORMAT_VERSION, "config": config,
                "feature_names": list(feature_names), "training": training_info,
                "versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "catboost", "lightgbm")}}
    with tempfile.TemporaryDirectory(prefix="hw3-model-") as directory:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
            for i, (model, component) in enumerate(zip(models, config["components"])):
                name = f"model_{i}." + ("cbm" if component["family"] == "catboost" else "txt")
                filename = Path(directory) / name
                model.save_model(str(filename))
                archive.write(filename, name)


def load_models(path):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported model archive version")
        models = []
        with tempfile.TemporaryDirectory(prefix="hw3-load-") as directory:
            for i, component in enumerate(metadata["config"]["components"]):
                if component["family"] == "catboost":
                    from catboost import CatBoostRanker
                    name = f"model_{i}.cbm"
                    filename = Path(directory) / name
                    filename.write_bytes(archive.read(name))
                    model = CatBoostRanker().load_model(str(filename))
                elif component["family"] == "lightgbm":
                    import lightgbm as lgb
                    model = lgb.Booster(model_str=archive.read(f"model_{i}.txt").decode())
                else:
                    raise ValueError("Unknown family in model archive")
                models.append(model)
    return models, metadata
