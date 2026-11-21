#!/usr/bin/env python3
"""Pinned, local multilingual cross-encoder scores with resumable checkpoints.

The network is used as released, without fitting to any VK MARCO labels. One
deterministic query-centred body window plus the title is encoded per pair.
Partial runs preserve the original pair-row indices: uncomputed scores are NaN,
and consumers must check ``mask`` instead of treating missing scores as zero.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from bisect import bisect_left

import numpy as np

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
WEIGHTS_SHA256 = "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer_config.json",
               "special_tokens_map.json", "tokenizer.json", "sentencepiece.bpe.model")
VERSION = "query-window-v1"
WORDS = re.compile(r"(?u)\w+")
# Only passage selection uses these stop words; the model receives original text.
STOP = frozenset("и в во на по с со из от до для к ко у а но или не это как что где когда какой какая какие ли чем the a an of to in on for and or is".split())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def download_model(model_dir: Path) -> dict:
    """Explicitly fetch only the immutable revision's non-pickle model assets."""
    from huggingface_hub import snapshot_download
    model_dir = Path(model_dir)
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir,
                      allow_patterns=list(MODEL_FILES) + ["README.md"])
    manifest = model_manifest(model_dir)
    (model_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def model_manifest(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing local model assets: {missing}; use --download_model explicitly")
    hashes = {name: file_sha256(model_dir / name) for name in MODEL_FILES}
    if hashes["model.safetensors"] != WEIGHTS_SHA256:
        raise ValueError("The model weights do not match the pinned public revision")
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "license": "Apache-2.0", "files_sha256": hashes,
            "pretraining_provenance": "Author reports multilingual machine-translated MS MARCO; not fitted here"}


@lru_cache(maxsize=1)
def _stemmer():
    import Stemmer
    return Stemmer.Stemmer("russian")


def _terms(words: list[str]) -> list[str]:
    return _stemmer().stemWords([w.lower().replace("ё", "е") for w in words])


def choose_body_window(query: str, body: str, window_words: int = 110,
                       stride: int = 55) -> tuple[str, int, int]:
    """Choose one fixed-length window by query-stem coverage, no labels/IDF.

    Candidate starts are a fixed overlapping grid plus the final full window.
    We maximize distinct non-stop query stems, then sum(log(1+term frequency)),
    then choose the earliest window. Empty-match queries use the document head.
    Returns original body text and word offsets (end exclusive).
    """
    if window_words < 1 or stride < 1 or stride > window_words:
        raise ValueError("Require 1 <= stride <= window_words")
    spans = list(WORDS.finditer(body))
    if not spans:
        return "", 0, 0
    n = len(spans)
    if n <= window_words:
        return body, 0, n
    raw_query = [w.lower().replace("ё", "е") for w in WORDS.findall(query)]
    wanted = set(_terms([w for w in raw_query if w not in STOP and len(w) > 1]))
    if not wanted:
        wanted = set(_terms(raw_query))
    positions = {term: [] for term in wanted}
    for i, word in enumerate(_terms([m.group() for m in spans])):
        if word in positions:
            positions[word].append(i)
    starts = list(range(0, n - window_words + 1, stride))
    if starts[-1] != n - window_words:
        starts.append(n - window_words)
    best_start, best_key = 0, (-1, -1.0)
    for start in starts:
        counts = [bisect_left(p, start + window_words) - bisect_left(p, start)
                  for p in positions.values()]
        key = (sum(v > 0 for v in counts), sum(math.log1p(v) for v in counts))
        if key > best_key:
            best_start, best_key = start, key
    end = best_start + window_words
    return body[spans[best_start].start():spans[end - 1].end()], best_start, end


def make_passage(query: str, title: str, body: str, window_words: int = 110,
                 stride: int = 55, title_words: int = 40) -> str:
    if title_words < 0:
        raise ValueError("title_words must be nonnegative")
    passage, _, _ = choose_body_window(query, body, window_words, stride)
    title = " ".join(title.split()[:title_words])
    return (title + "\n" + passage).strip()


def pair_fingerprint(pairs) -> str:
    """Length-prefixed hashes avoid ambiguity, and include query text, not labels."""
    h = hashlib.sha256()
    for qid, did, oid in zip(pairs.query_ids, pairs.doc_ids, pairs.object_ids, strict=True):
        for value in (str(qid), str(did), str(oid)):
            data = value.encode("utf-8")
            h.update(len(data).to_bytes(8, "little")); h.update(data)
    for qid, text in sorted(pairs.query_text.items()):
        for value in (str(qid), str(text)):
            data = value.encode("utf-8")
            h.update(len(data).to_bytes(8, "little")); h.update(data)
    return h.hexdigest()


def select_rows(pairs, partition="all", query_ids=None, query_limit=None,
                rows=None) -> np.ndarray:
    count = len(pairs.query_ids)
    selected = np.ones(count, dtype=bool)
    if partition == "train":
        selected &= np.asarray(pairs.is_train, dtype=bool)
    elif partition == "test":
        selected &= ~np.asarray(pairs.is_train, dtype=bool)
    elif partition != "all":
        raise ValueError("partition must be all, train or test")
    qids = np.asarray(pairs.query_ids).astype(str)
    if query_ids is not None:
        unknown = set(map(str, query_ids)) - set(qids)
        if unknown:
            raise ValueError(f"Unknown prefixed QueryIds: {sorted(unknown)[:3]}")
        selected &= np.isin(qids, list(map(str, query_ids)))
    if query_limit is not None:
        if query_limit < 1:
            raise ValueError("query_limit must be positive")
        selected &= np.isin(qids, np.unique(qids[selected])[:query_limit])
    if rows is not None:
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if rows.shape != (count,):
                raise ValueError("Boolean rows_file must match the complete pair count")
            selected &= rows
        else:
            if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
                raise ValueError("rows_file must contain a boolean mask or integer row indices")
            if np.any(rows < 0) or np.any(rows >= count) or len(np.unique(rows)) != len(rows):
                raise ValueError("Row indices must be unique and in range")
            mask = np.zeros(count, dtype=bool); mask[rows] = True
            selected &= mask
    return np.flatnonzero(selected)


def topk_rows(query_ids, baseline_scores, k, tie_keys=None) -> np.ndarray:
    """Label-free optional cascade selector, independent within each query."""
    qids = np.asarray(query_ids).astype(str)
    scores = np.asarray(baseline_scores, dtype=float)
    if k < 1 or scores.shape != qids.shape or not np.isfinite(scores).all():
        raise ValueError("Top-k selection needs finite scores and positive k")
    ties = np.arange(len(qids)) if tie_keys is None else np.asarray(tie_keys)
    if ties.shape != qids.shape:
        raise ValueError("tie_keys must match query_ids")
    if not len(qids):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((ties, -scores, qids))
    sorted_qids = qids[order]
    positions = np.arange(len(order))
    starts = np.maximum.accumulate(np.where(np.r_[True, sorted_qids[1:] != sorted_qids[:-1]], positions, 0))
    return np.sort(order[(positions - starts) < k])


def save_checkpoint(path, scores, mask, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".neural-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(out, scores=np.asarray(scores, dtype=np.float32),
                                mask=np.asarray(mask, dtype=bool),
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(path, count, signature):
    if not Path(path).exists():
        return np.full(count, np.nan, dtype=np.float32), np.zeros(count, dtype=bool), {}
    with np.load(path, allow_pickle=False) as data:
        scores = data["scores"].astype(np.float32)
        mask = data["mask"].astype(bool)
        meta = json.loads(str(data["metadata"]))
    if scores.shape != (count,) or mask.shape != (count,) or meta.get("signature") != signature:
        raise ValueError("Stale neural checkpoint: pair order, model, source texts or inference configuration changed")
    if not np.isfinite(scores[mask]).all() or not np.isnan(scores[~mask]).all():
        raise ValueError("Corrupt neural checkpoint mask/scores")
    return scores, mask, meta


class LocalCrossEncoder:
    def __init__(self, model_dir, device="cpu", max_length=256, threads=4):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be cpu, mps or cuda")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this machine")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine")
        if not 32 <= max_length <= 512 or threads < 1:
            raise ValueError("Require 32 <= max_length <= 512 and threads >= 1")
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        self.torch, self.device, self.max_length = torch, device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                                      trust_remote_code=False, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True).to(device).eval()
        if self.model.config.num_labels != 1:
            raise ValueError("The pinned cross-encoder must return a single relevance logit")
        self._short_query = lru_cache(maxsize=20000)(self._truncate_query)

    def _truncate_query(self, query):
        limit = min(64, self.max_length // 3)
        tokens = self.tokenizer.encode(query, add_special_tokens=False)
        return query if len(tokens) <= limit else self.tokenizer.decode(tokens[:limit], skip_special_tokens=True)

    def predict(self, queries, passages) -> np.ndarray:
        if len(queries) != len(passages):
            raise ValueError("Each query must have one passage")
        if not queries:
            return np.empty(0, dtype=np.float32)
        encoded = self.tokenizer([self._short_query(q) for q in queries], passages,
                                 padding=True, truncation="only_second",
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            result = self.model(**encoded).logits[:, 0].float().cpu().numpy()
        if result.shape != (len(queries),) or not np.isfinite(result).all():
            raise ValueError("The neural model produced invalid scores")
        return result


def score_pairs(pairs, database, model_dir, output_file, selected_rows, *, device="cpu",
                batch_size=16, max_length=256, window_words=110, stride=55,
                title_words=40, threads=4, checkpoint_batches=50):
    if batch_size < 1 or checkpoint_batches < 1:
        raise ValueError("batch_size and checkpoint_batches must be positive")
    database = Path(database)
    if not database.is_file():
        raise ValueError(f"Document cache is missing: {database}")
    selected_rows = select_rows(pairs, rows=np.asarray(selected_rows))
    manifest = model_manifest(Path(model_dir))
    # SQLite must be final/closed by the builder before a neural run. Changed DB
    # mtime/size invalidates all partial scores, rather than silently mixing data.
    source = {"database_size": database.stat().st_size,
              "database_mtime_ns": database.stat().st_mtime_ns,
              "pair_fingerprint": pair_fingerprint(pairs)}
    config = {"version": VERSION, "implementation_sha256": file_sha256(Path(__file__)),
              "max_length": max_length, "window_words": window_words,
              "stride": stride, "title_words": title_words, "query_token_cap": min(64, max_length // 3),
              "device": device, "precision": "float32", "batch_size": batch_size,
              "passage": "title followed by one query-stem coverage body window"}
    signature = _json_hash({"source": source, "model": manifest, "config": config})
    scores, mask, previous = load_checkpoint(output_file, len(pairs.query_ids), signature)
    pending = selected_rows[~mask[selected_rows]]
    metadata = {"signature": signature, "model": manifest, "source": source,
                "config": config, "requested_rows": len(selected_rows),
                "requested_rows_sha256": hashlib.sha256(selected_rows.astype('<i8').tobytes()).hexdigest(),
                "runs": list(previous.get("runs", []))}
    if not len(pending):
        metadata.update(computed_rows=int(mask.sum()), requested_complete=True)
        save_checkpoint(output_file, scores, mask, metadata)
        print(f"All {len(selected_rows)} requested rows are already cached", flush=True)
        return scores, mask, metadata
    started = time.perf_counter()
    encoder = LocalCrossEncoder(model_dir, device, max_length, threads)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    completed = 0
    try:
        for batch_index, offset in enumerate(range(0, len(pending), batch_size), 1):
            rows = pending[offset:offset + batch_size]
            queries, passages = [], []
            for row in rows:
                document = connection.execute("SELECT title,body FROM documents WHERE doc_id=?",
                                              (str(pairs.doc_ids[row]),)).fetchone()
                if document is None:
                    raise ValueError(f"Document missing from cache: {pairs.doc_ids[row]}")
                query = pairs.query_text[str(pairs.query_ids[row])]
                queries.append(query)
                passages.append(make_passage(query, document[0] or "", document[1] or "",
                                             window_words, stride, title_words))
            scores[rows] = encoder.predict(queries, passages)
            mask[rows] = True
            completed += len(rows)
            if batch_index % checkpoint_batches == 0 or completed == len(pending):
                elapsed = time.perf_counter() - started
                metadata.update(computed_rows=int(mask.sum()), requested_complete=bool(mask[selected_rows].all()),
                                latest_run={"rows": completed, "seconds": elapsed,
                                            "pairs_per_second": completed / elapsed})
                save_checkpoint(output_file, scores, mask, metadata)
                print(f"neural completed={completed}/{len(pending)}, total_cached={mask.sum()}, "
                      f"elapsed={elapsed:.1f}s, pairs/s={completed/elapsed:.2f}", flush=True)
    except BaseException:
        metadata.update(computed_rows=int(mask.sum()), requested_complete=False)
        save_checkpoint(output_file, scores, mask, metadata)
        raise
    finally:
        connection.close()
    elapsed = time.perf_counter() - started
    metadata["runs"].append({"rows": completed, "seconds": elapsed,
                             "pairs_per_second": completed / elapsed})
    save_checkpoint(output_file, scores, mask, metadata)
    return scores, mask, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--window_words", type=int, default=110)
    parser.add_argument("--stride", type=int, default=55)
    parser.add_argument("--title_words", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint_batches", type=int, default=50)
    parser.add_argument("--partition", choices=["all", "train", "test"], default="all")
    parser.add_argument("--query_limit", type=int)
    parser.add_argument("--query_ids_file", type=Path, help="JSON list or one prefixed QueryId per line")
    parser.add_argument("--rows_file", type=Path, help="NPY of integer pair-row indices or bool mask")
    parser.add_argument("--download_model", action="store_true")
    args = parser.parse_args()
    from data import load_pairs
    if args.output_file.resolve() == (args.cache_dir / "features.npz").resolve():
        parser.error("Neural output must not overwrite features.npz")
    if args.output_file.resolve() == (args.cache_dir / "documents.sqlite").resolve():
        parser.error("Neural output must not overwrite the document cache")
    if any(args.output_file.resolve() == p.resolve() for p in args.data_dir.iterdir() if p.is_file()):
        parser.error("Neural output must not overwrite source data")
    try:
        if args.download_model:
            download_model(args.model_dir)
        pairs = load_pairs(args.data_dir)
        query_ids = None
        if args.query_ids_file:
            text = args.query_ids_file.read_text(encoding="utf-8")
            query_ids = json.loads(text) if text.lstrip().startswith("[") else text.splitlines()
        rows = np.load(args.rows_file, allow_pickle=False) if args.rows_file else None
        selected = select_rows(pairs, args.partition, query_ids, args.query_limit, rows)
        if not len(selected):
            raise ValueError("The selectors matched no candidate pairs")
        score_pairs(pairs, args.cache_dir / "documents.sqlite", args.model_dir,
                    args.output_file, selected, device=args.device, batch_size=args.batch_size,
                    max_length=args.max_length, window_words=args.window_words, stride=args.stride,
                    title_words=args.title_words, threads=args.threads,
                    checkpoint_batches=args.checkpoint_batches)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
