#!/usr/bin/env python3
"""HW3: train a CPU ranker, or load it and score the supplied test candidates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter

import pandas as pd

from data import load_test, load_train
from modeling import ensemble_predict, fit_component, load_models, save_models

ROOT = Path(__file__).resolve().parent


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def temporary_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False)
    handle.close()
    return Path(handle.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_file", type=Path, default=Path("model.bin"))
    parser.add_argument("--submission_file", type=Path, default=Path("submission.csv"))
    parser.add_argument("--config_file", type=Path, default=ROOT / "best_config.json",
                        help="Fixed final configuration; used only in training mode")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    started = perf_counter()
    output = args.model_file.resolve() if args.train else args.submission_file.resolve()
    protected = {(args.data_dir / name).resolve() for name in
                 ("train.csv", "test.csv", "sample_submission.csv", "features.md")}
    protected.add(args.config_file.resolve())
    if not args.train:
        protected.add(args.model_file.resolve())
    if output in protected:
        parser.error("An output must not overwrite an input, config, or loaded model")
    temporary = None
    try:
        if args.train:
            config = json.loads(args.config_file.read_text(encoding="utf-8"))
            dataset = load_train(args.data_dir / "train.csv")
            models, components = [], []
            for component in config["components"]:
                model, fitted_config = fit_component(dataset, component,
                                                     seed=int(component.get("seed", config["seed"])), threads=args.threads)
                models.append(model)
                components.append(fitted_config)
            config = dict(config, components=components)
            info = {"rows": len(dataset.query_ids), "queries": len(set(dataset.query_ids)),
                    "train_sha256": digest(args.data_dir / "train.csv"),
                    "threads": args.threads, "seconds_before_save": perf_counter() - started}
            temporary = temporary_output(output)
            save_models(temporary, models, config, dataset.feature_names, info)
        else:
            models, metadata = load_models(args.model_file)
            dataset = load_test(args.data_dir / "test.csv", tuple(metadata["feature_names"]))
            scores = ensemble_predict(models, dataset, metadata["config"], threads=args.threads)
            temporary = temporary_output(output)
            pd.DataFrame({"ObjectId": dataset.object_ids, "Score": scores}).to_csv(
                temporary, index=False, float_format="%.17g", lineterminator="\n")
        os.replace(temporary, output)
        temporary = None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_bytes = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
        print(json.dumps({"mode": "train" if args.train else "predict", "output": str(output),
                          "rows": len(dataset.query_ids), "queries": len(set(dataset.query_ids)),
                          "seconds": round(perf_counter() - started, 3),
                          "peak_rss_bytes": peak_bytes, "sha256": digest(output)}, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
