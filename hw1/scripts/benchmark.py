#!/usr/bin/env python3
"""Measure each solution process separately, including imports (Linux/macOS)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def measure(command, log):
    started = perf_counter()
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    seconds = perf_counter() - started
    peak_bytes = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    metrics = {"command": command, "seconds": round(seconds, 4),
               "peak_rss_bytes": peak_bytes, "peak_rss_mib": round(peak_bytes / 2**20, 3),
               "returncode": process.returncode, "log": log.name}
    print(json.dumps(metrics), flush=True)
    if process.returncode:
        raise RuntimeError(f"Command failed; see {log}")
    return metrics


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["synthetic", "official"], required=True)
    parser.add_argument("--submission-only", action="store_true")
    parser.add_argument("--index_dir", type=Path)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    if args.kind == "official" and (data / "synthetic.json").exists():
        parser.error("synthetic data must not be reported as official")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    index = args.index_dir.resolve() if args.index_dir else output / "index"
    submission = output / "submission.csv"
    results = {"kind": args.kind, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
               "platform": platform.platform(), "python": sys.version, "data_dir": str(data),
               "method": "wait4 ru_maxrss per fresh child; wall time includes interpreter/imports",
               "solution_sha256": digest(ROOT / "solution.py"),
               "requirements_sha256": digest(ROOT / "requirements.txt")}
    if not args.submission_only:
        results["build"] = measure([sys.executable, str(ROOT / "solution.py"), "--build_index",
                                    f"--index_dir={index}", str(data)], output / "build.log")
    results["submission"] = measure([sys.executable, str(ROOT / "solution.py"),
                                     f"--submission_file={submission}", f"--index_dir={index}",
                                     str(data)], output / "submission.log")
    results["index_bytes"] = sum(path.stat().st_size for path in index.rglob("*") if path.is_file())
    results["index_mib"] = round(results["index_bytes"] / 2**20, 3)
    connection = sqlite3.connect((index / "index.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        results["index_metadata"] = dict(connection.execute("SELECT key, value FROM metadata"))
    finally:
        connection.close()
    results["submission_sha256"] = digest(submission)
    # Every byte of the synthetic output is checked, not just a sample of rows.
    expected = data / "expected_submission.csv"
    if expected.is_file():
        results["matches_expected"] = results["submission_sha256"] == digest(expected)
    results["submission_rss_below_100_mb"] = results["submission"]["peak_rss_bytes"] <= 100_000_000
    (output / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if results.get("matches_expected") is False:
        raise SystemExit("ERROR: submission differs from exact synthetic labels")
    if not results["submission_rss_below_100_mb"]:
        raise SystemExit("ERROR: submission peak RSS exceeds 100,000,000 bytes")


if __name__ == "__main__":
    main()
