#!/usr/bin/env python3
"""Run inside a fresh Linux cgroup v2 container with a 100 MB cap and no swap."""

import argparse
import json
from pathlib import Path
import platform
import sys

from benchmark import ROOT, digest, measure


def read_events(path):
    return {name: int(value) for name, value in
            (line.split() for line in path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    cgroup = Path("/sys/fs/cgroup")
    cap_path = cgroup / "memory.max"
    if not cap_path.is_file() or cap_path.read_text().strip() == "max":
        parser.error("requires cgroup v2 with --memory=100000000 --memory-swap=100000000")
    limit = int(cap_path.read_text())
    if limit > 100_000_000 or int((cgroup / "memory.swap.max").read_text()) != 0:
        parser.error("use a limit of at most 100000000 bytes and disable swap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    submission = output / "submission.csv"
    before = read_events(cgroup / "memory.events")
    measured = measure([sys.executable, str(ROOT / "solution.py"),
                        f"--submission_file={submission}",
                        f"--index_dir={args.index_dir.resolve()}", str(args.data_dir.resolve())],
                       output / "submission.log")
    after = read_events(cgroup / "memory.events")
    result = {"platform": platform.platform(), "python": sys.version,
              "kind": "synthetic" if (args.data_dir / "synthetic.json").exists() else "unverified",
              "limit_bytes": limit, "swap_limit_bytes": 0,
              "cgroup_peak_bytes": int((cgroup / "memory.peak").read_text()),
              "events_delta": {name: after[name] - before.get(name, 0) for name in after},
              "solution_sha256": digest(ROOT / "solution.py"),
              "requirements_sha256": digest(ROOT / "requirements.txt"),
              "submission": measured, "submission_sha256": digest(submission)}
    expected = args.data_dir / "expected_submission.csv"
    if expected.is_file():
        result["matches_expected"] = result["submission_sha256"] == digest(expected)
    result["passed"] = (measured["returncode"] == 0 and result.get("matches_expected", True)
                        and result["events_delta"].get("oom_kill", 0) == 0
                        and result["events_delta"].get("oom", 0) == 0)
    (output / "memory-limit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Memory limit check failed")


if __name__ == "__main__":
    main()
