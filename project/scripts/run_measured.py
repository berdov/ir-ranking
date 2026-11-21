#!/usr/bin/env python3
"""Run a command and record wall time and sampled process-tree RSS."""
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command:
    parser.error('A command after -- is required')
started = time.monotonic()
process = subprocess.Popen(command)
observed = psutil.Process(process.pid)
peak = 0
while process.poll() is None:
    try:
        members = [observed] + observed.children(recursive=True)
        total = 0
        for member in members:
            try:
                total += member.memory_info().rss
            except psutil.Error:
                pass
        peak = max(peak, total)
    except psutil.Error:
        pass
    time.sleep(.25)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
native_peak = usage.ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
report = dict(command=command, exit_code=process.returncode,
              wall_seconds=time.monotonic()-started,
              sampled_process_tree_peak_rss_bytes=peak,
              native_child_peak_rss_bytes=native_peak,
              rss_excludes_separate_gpu_memory=True,
              platform=platform.platform(), machine=platform.machine(),
              python=platform.python_version())
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
raise SystemExit(process.returncode)
