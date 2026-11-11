#!/usr/bin/env python3
"""Measure HW3 training and inference in separate fresh CPU processes.

Training must finish in 1,800 seconds and use at most 8 GB of RAM. The same
limits are conservatively applied to inference. Both include Python startup and
imports. Exact child peak RSS comes from wait4; Linux cgroup v2 can enforce a
hard memory ceiling with no swap. This script only reads HW3 inputs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from time import perf_counter, sleep

from validate_submission import validate, write_json


ROOT = Path(__file__).resolve().parents[1]
TIME_LIMIT_SECONDS = 1800.0
MEMORY_LIMIT_BYTES = 8_000_000_000
DATA_FILES = ("train.csv", "test.csv", "features.md", "sample_submission.csv")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def cgroup_state():
    """Inspect cgroup v2 as mounted at the root of a fresh Docker container."""
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
    """Supplementary live sample; the final result uses the exact wait4 peak."""
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
        pass
    return None


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def measure(command, log, time_limit=TIME_LIMIT_SECONDS, memory_limit=MEMORY_LIMIT_BYTES):
    if sys.platform not in {"linux", "darwin"}:
        raise ValueError("Peak RSS measurement requires Linux or macOS wait4")
    started, sampled_peak, reason = perf_counter(), 0, None
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
                sampled_peak = max(sampled_peak, current_rss(process.pid) or 0)
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
    peak_bytes = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    within_time, within_memory = seconds <= time_limit, peak_bytes <= memory_limit
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
    parser.add_argument("--config_file", type=Path, default=ROOT / "best_config.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--require-cgroup", action="store_true",
                        help="Require Linux cgroup v2 memory <=8 GB with swap disabled")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    config = args.config_file.resolve()
    outputs = {name: output_dir / name for name in
               ("model.bin", "submission.csv", "train.log", "predict.log",
                "metrics.json", "validation.json")}
    source_paths = [*ROOT.glob("*.py"), ROOT / "requirements.txt", ROOT / "best_config.json",
                    *Path(__file__).parent.glob("*.py")]
    protected = {(data_dir / name).resolve() for name in DATA_FILES}
    protected.update(path.resolve() for path in source_paths)
    protected.add(config)
    resolved_outputs = [path.resolve() for path in outputs.values()]
    if len(set(resolved_outputs)) != len(resolved_outputs) or any(
            path in protected for path in resolved_outputs):
        parser.error("Benchmark outputs must be distinct and cannot overwrite data, code, or config")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": args.kind, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version, "data_dir": str(data_dir),
        "training_time_limit_seconds": TIME_LIMIT_SECONDS,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES, "threads": args.threads,
        "method": "Fresh child per mode; wall time includes imports; exact wait4 peak RSS",
        "memory_enforcement": "Live RSS monitoring and exact post-run wait4 peak check",
        "passed": False, "public_kaggle_score": None,
    }
    try:
        report["source_sha256"] = {str(path.relative_to(ROOT)): digest(path)
                                   for path in sorted(set(source_paths))}
        report["config"] = {"path": str(config), "sha256": digest(config)}
        report["input_files"] = {}
        for name in DATA_FILES:
            path = data_dir / name
            if not path.is_file():
                if name in {"train.csv", "test.csv"} or args.kind == "official":
                    raise ValueError(f"Missing required HW3 input: {name}")
                continue
            report["input_files"][name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        before = cgroup_state()
        report["cgroup_before"] = before
        if args.require_cgroup:
            limit = before.get("memory.max") if before else None
            swap = before.get("memory.swap.max") if before else None
            if not isinstance(limit, int) or limit > MEMORY_LIMIT_BYTES or swap != 0:
                raise ValueError("Use a fresh Linux container with --memory=8000000000 "
                                 "--memory-swap=8000000000; cgroup v2 is required")
            report["memory_enforcement"] = "Linux cgroup v2 hard cap, swap disabled, plus wait4 RSS"
        # Remove only named benchmark products after all input/output checks.
        for name in ("model.bin", "submission.csv", "validation.json"):
            outputs[name].unlink(missing_ok=True)
        common = [sys.executable, str(ROOT / "solution.py"),
                  f"--model_file={outputs['model.bin']}", f"--threads={args.threads}"]
        report["train"] = measure(
            [*common, "--train", f"--config_file={config}", str(data_dir)], outputs["train.log"])
        if report["train"]["passed"] and outputs["model.bin"].is_file():
            report["model_sha256"] = digest(outputs["model.bin"])
            report["predict"] = measure(
                [*common, f"--submission_file={outputs['submission.csv']}", str(data_dir)],
                outputs["predict.log"])
            if report["predict"]["passed"] and outputs["submission.csv"].is_file():
                report["validation"] = validate(data_dir, outputs["submission.csv"])
                write_json(outputs["validation.json"], report["validation"])
                report["submission_sha256"] = report["validation"]["submission_sha256"]
        after = cgroup_state()
        report["cgroup_after"] = after
        events = {}
        if before and after:
            events = {name: value - before.get("events", {}).get(name, 0)
                      for name, value in after.get("events", {}).items()}
        report["cgroup_events_delta"] = events
        report["passed"] = bool(report["train"]["passed"]
                                and report.get("predict", {}).get("passed", False)
                                and report.get("validation", {}).get("passed", False)
                                and not events.get("oom", 0) and not events.get("oom_kill", 0))
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        report["error"] = str(error)
    write_json(outputs["metrics.json"], report)
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
