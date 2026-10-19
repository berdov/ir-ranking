#!/usr/bin/env python3
"""Measure one complete HW2 solution run on Linux/macOS, including imports.

The 30-minute deadline is actively enforced. Peak RSS comes from wait4, not
from periodic samples. Samples additionally stop a running process that exceeds
8 GB; --require-cgroup verifies a kernel memory cap and disabled swap on Linux.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
from time import perf_counter, sleep


ROOT = Path(__file__).resolve().parents[1]
TIME_LIMIT_SECONDS = 1800.0
MEMORY_LIMIT_BYTES = 8_000_000_000
DATA_FILES = (
    "vkmarco-docs.tsv", "vkmarco-doctrain-queries.tsv", "vkmarco-doctrain-qrels.tsv",
    "vkmarco-doceval-queries.tsv", "objects.csv", "sample_submission.csv",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cgroup_state():
    """Read the current v2 cgroup, when mounted at the container's root."""
    root = Path("/sys/fs/cgroup")
    if sys.platform != "linux" or not (root / "memory.max").is_file():
        return None
    result = {"path": str(root)}
    for name in ("memory.max", "memory.swap.max", "memory.peak"):
        path = root / name
        if path.is_file():
            value = path.read_text().strip()
            result[name] = int(value) if value.isdigit() else value
    events = root / "memory.events"
    if events.is_file():
        result["events"] = {name: int(value) for name, value in
                            (line.split() for line in events.read_text().splitlines())}
    return result


def current_rss(pid):
    """Supplemental live RSS sample; final acceptance uses exact wait4 peak."""
    try:
        if sys.platform == "linux":
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            result = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(pid)],
                                    capture_output=True, text=True, timeout=1)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass  # A process may exit between wait4(WNOHANG) and this sample.
    return None


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def measure(command, log, time_limit=TIME_LIMIT_SECONDS, memory_limit=MEMORY_LIMIT_BYTES):
    """Return metrics even when the child fails, times out, or exceeds RSS."""
    if sys.platform not in {"linux", "darwin"}:
        raise ValueError("RSS measurement requires Linux or macOS wait4")
    started = perf_counter()
    reason = None
    sampled_peak = 0
    with Path(log).open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                finished, status, usage = os.wait4(process.pid, os.WNOHANG)
                if finished:
                    break
                if perf_counter() - started > time_limit:
                    reason = "time_limit"
                    kill_group(process)
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                sample = current_rss(process.pid)
                sampled_peak = max(sampled_peak, sample or 0)
                if sampled_peak > memory_limit:
                    reason = "memory_limit"
                    kill_group(process)
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                sleep(0.05)
        except BaseException:
            kill_group(process)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            raise
        process.returncode = os.waitstatus_to_exitcode(status)
    seconds = perf_counter() - started
    peak_bytes = int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024)
    within_time = seconds <= time_limit
    within_memory = peak_bytes <= memory_limit
    return {
        "command": [str(part) for part in command], "seconds": round(seconds, 4),
        "peak_rss_bytes": peak_bytes, "peak_rss_mib": round(peak_bytes / 2**20, 3),
        "sampled_peak_rss_bytes": sampled_peak, "returncode": process.returncode,
        "termination_reason": reason, "within_time_limit": within_time,
        "within_memory_limit": within_memory, "log": str(log),
        "passed": process.returncode == 0 and within_time and within_memory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("synthetic", "official"), required=True)
    parser.add_argument("--normalization", choices=("none", "lemma"), default="lemma")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cgroup", action="store_true",
                        help="Require a Linux cgroup v2 memory cap <=8 GB with swap disabled")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": args.kind, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version, "data_dir": str(data_dir),
        "time_limit_seconds": TIME_LIMIT_SECONDS, "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "method": "Fresh solution child; wall time includes interpreter/imports; peak RSS from wait4",
        "memory_enforcement": "Periodic RSS monitoring plus exact post-run wait4 peak check",
        "passed": False, "public_kaggle_score": None,
    }
    try:
        report["source_sha256"] = {name: digest(ROOT / name) for name in
                                   ("solution.py", "ranking.py", "requirements.txt", "scripts/benchmark.py")}
        report["input_files"] = {name: {"bytes": (data_dir / name).stat().st_size,
                                        "sha256": digest(data_dir / name)} for name in DATA_FILES}
        before = cgroup_state()
        report["cgroup_before"] = before
        if args.require_cgroup:
            limit = before.get("memory.max") if before else None
            swap = before.get("memory.swap.max") if before else None
            if not isinstance(limit, int) or limit > MEMORY_LIMIT_BYTES or swap != 0:
                raise ValueError("Use a fresh Linux container with --memory=8000000000 "
                                 "--memory-swap=8000000000; cgroup v2 is required")
            report["memory_enforcement"] = "Linux cgroup v2 hard cap, no swap; RSS also checked with wait4"
        submission = output_dir / "submission.csv"
        ranking_report = output_dir / "ranking.json"
        # A failed rerun must not leave an old output that looks like this run.
        protected = {(data_dir / name).resolve() for name in DATA_FILES}
        for path in (submission, ranking_report, output_dir / "submission.log", output_dir / "metrics.json"):
            if path.resolve() in protected:
                raise ValueError("Benchmark output must not overwrite a competition input file")
        submission.unlink(missing_ok=True)
        ranking_report.unlink(missing_ok=True)
        command = [sys.executable, str(ROOT / "solution.py"), f"--submission_file={submission}",
                   f"--report_file={ranking_report}", f"--normalization={args.normalization}",
                   "--seed", str(args.seed), str(data_dir)]
        measured = measure(command, output_dir / "submission.log")
        report["submission"] = measured
        after = cgroup_state()
        report["cgroup_after"] = after
        events = {}
        if before and after:
            events = {name: value - before.get("events", {}).get(name, 0)
                      for name, value in after.get("events", {}).items()}
        report["cgroup_events_delta"] = events
        report["passed"] = bool(measured["passed"] and not events.get("oom", 0)
                                and not events.get("oom_kill", 0) and submission.is_file()
                                and ranking_report.is_file())
        if submission.is_file():
            report["submission_sha256"] = digest(submission)
        if ranking_report.is_file():
            report["ranking_report"] = str(ranking_report)
    except (OSError, ValueError, RuntimeError) as error:
        report["error"] = str(error)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir,
                                     prefix=".metrics-", suffix=".json", delete=False) as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary_report = Path(handle.name)
    os.replace(temporary_report, output_dir / "metrics.json")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
