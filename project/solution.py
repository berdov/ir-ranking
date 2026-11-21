#!/usr/bin/env python3
"""Reproduce the final project submission from the six competition files.

    ./solution.py --submission_file=submission.csv PATH-TO-DATA

The frozen training-only configuration is shipped next to this script. The
first run extracts corpus features and refits the selected ranker. If selected,
the pinned public neural weights are downloaded once and used locally.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from data import FILE_NAMES, file_hash, load_pairs
from experiments import fit_final, predict
from features import build_features
from fusion import fuse_scores, required_rows
from portability import adapt_selection_for_runtime
from ranking import json_hash, subset, write_json

ROOT = Path(__file__).resolve().parent


def write_submission(path, object_ids, scores):
    path = Path(path)
    object_ids = np.asarray(object_ids).astype(str)
    scores = np.asarray(scores, dtype=np.float64)
    if object_ids.ndim != 1 or scores.shape != object_ids.shape:
        raise ValueError("One score per ObjectId is required")
    if not np.isfinite(scores).all() or len(np.unique(object_ids)) != len(object_ids):
        raise ValueError("Scores must be finite and ObjectIds unique")
    if not len(object_ids) or np.any(object_ids == ""):
        raise ValueError("Submission ObjectIds cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix='.submission-', delete=False) as out:
            temporary = Path(out.name)
            writer = csv.writer(out, lineterminator='\n')
            writer.writerow(['ObjectId', 'Score'])
            for oid, score in zip(object_ids, scores, strict=True):
                writer.writerow([str(oid), format(float(score), '.17g')])
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args):
    started = perf_counter()
    selection = json.loads(args.selection_file.read_text(encoding='utf-8'))
    if selection.get('quick_smoke_only'):
        raise ValueError('A smoke-test configuration is not a final submission model')
    fusion = json.loads(args.fusion_file.read_text(encoding='utf-8'))
    if fusion.get('lexical_selection_sha256', json_hash(selection)) != json_hash(selection):
        raise ValueError('Fusion was selected for a different lexical model')
    config = fusion['config'] if not args.lexical_only else {'neural_weight': 0., 'top_k': 0}
    protected = {(args.data_dir/name).resolve() for name in FILE_NAMES}
    protected |= {args.selection_file.resolve(), args.fusion_file.resolve(), args.portability_file.resolve(),
                  (args.cache_dir/'features.npz').resolve(), (args.cache_dir/'documents.sqlite').resolve()}
    if args.submission_file.resolve() in protected or args.report_file.resolve() in protected:
        raise ValueError('Output files must not overwrite inputs, configuration or caches')
    if args.submission_file.resolve() == args.report_file.resolve():
        raise ValueError('Submission and report must have different paths')
    bundle = build_features(args.data_dir, args.cache_dir)
    if not bundle.metadata.get('complete'):
        raise ValueError('A partial profiling cache cannot produce a final submission')
    runtime_selection, portability_report = adapt_selection_for_runtime(
        bundle, selection, args.data_dir, args.portability_file)
    trained = fit_final(bundle, runtime_selection, args.model_dir, threads=args.threads)
    test_rows = np.flatnonzero(~bundle.is_train)
    test = subset(bundle, test_rows)
    lexical = predict(test, args.model_dir, threads=args.threads)
    needed = required_rows(test.query_ids, test.doc_ids, lexical, config)
    neural_report = None
    if needed.any():
        from neural import MODEL_FILES, MODEL_REVISION, download_model, score_pairs
        if fusion.get('neural', {}).get('revision', MODEL_REVISION) != MODEL_REVISION:
            raise ValueError('Fusion requires a different pinned neural model revision')
        if any(not (args.neural_model_dir/name).is_file() for name in MODEL_FILES):
            if args.offline:
                raise ValueError('Pinned neural weights are missing in offline mode')
            download_model(args.neural_model_dir)
        pairs = load_pairs(args.data_dir)
        for name in ('query_ids', 'doc_ids', 'object_ids'):
            if not np.array_equal(getattr(pairs, name), getattr(bundle, name)):
                raise ValueError('Cached features and neural pair order differ')
        device = args.device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        batch_size = args.batch_size or (32 if device != 'cpu' else 16)
        settings = fusion.get('neural', {})
        allowed = ('max_length', 'window_words', 'stride', 'title_words')
        kwargs = {name: settings[name] for name in allowed if name in settings}
        cache_name = f'neural-{device}-b{batch_size}-{json_hash(kwargs)[:10]}.npz'
        neural_scores, mask, neural_report = score_pairs(
            pairs, args.cache_dir/'documents.sqlite', args.neural_model_dir,
            args.cache_dir/cache_name, test_rows[needed], device=device,
            batch_size=batch_size, threads=args.threads, **kwargs)
        result = fuse_scores(test.query_ids, test.doc_ids, lexical,
                             neural_scores[test_rows], mask[test_rows], config)
    else:
        result = lexical
    write_submission(args.submission_file, test.object_ids, result)
    report = dict(submission=str(args.submission_file.resolve()),
                  submission_sha256=file_hash(args.submission_file),
                  rows=len(result), queries=len(np.unique(test.query_ids)),
                  score_min=float(result.min()), score_max=float(result.max()),
                  selection_sha256=json_hash(selection), fusion=config,
                  lexical_only=bool(args.lexical_only), neural_pairs=int(needed.sum()),
                  elapsed_seconds=perf_counter()-started,
                  features=bundle.metadata, training=trained, portability=portability_report,
                  neural=neural_report, public_kaggle_score=None)
    write_json(args.report_file, report)
    print(json.dumps({key: report[key] for key in ('submission', 'rows', 'queries',
                      'neural_pairs', 'elapsed_seconds', 'submission_sha256')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--submission_file', required=True, type=Path)
    parser.add_argument('--cache_dir', type=Path, default=ROOT/'cache')
    parser.add_argument('--model_dir', type=Path, default=ROOT/'models'/'final')
    parser.add_argument('--selection_file', type=Path, default=ROOT/'best_config.json')
    parser.add_argument('--fusion_file', type=Path, default=ROOT/'fusion_config.json')
    parser.add_argument('--portability_file', type=Path, default=ROOT/'portability_manifest.json')
    parser.add_argument('--neural_model_dir', type=Path, default=ROOT/'cache'/'neural-model')
    parser.add_argument('--report_file', type=Path, default=ROOT/'reports'/'reproduction.json')
    parser.add_argument('--device', choices=['cpu','mps','cuda','auto'], default='cpu')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--offline', action='store_true', help='Never download missing model weights')
    parser.add_argument('--lexical_only', action='store_true', help='Explicit lexical ablation, not the selected fusion')
    args = parser.parse_args()
    if args.threads < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('Threads and batch size must be positive')
    try:
        run(args)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
