#!/usr/bin/env python3
"""Package source, reproduction tools and small reports without data or environments."""

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "dist" / "hw1_berdov.zip")
    args = parser.parse_args()
    required = [ROOT / name for name in ("solution.py", "requirements.txt", "README.md")]
    for path in required:
        if not path.is_file():
            parser.error(f"missing required file: {path}")
    files = required + [ROOT / "Dockerfile", ROOT / ".dockerignore"]
    files += list((ROOT / "scripts").glob("*.py"))
    files += list((ROOT / "tests").glob("*.py"))
    for pattern in ("*/metrics.json", "*/build.log", "*/submission.log", "*/memory-limit.json",
                    "*/validation.json", "*/data-manifest.json", "*/kaggle-result.json"):
        files += list((ROOT / "reports").glob(pattern))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(args.output) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"Invalid ZIP member: {bad}")
        if not ((archive.getinfo("solution.py").external_attr >> 16) & 0o111):
            raise SystemExit("solution.py must be executable before packaging")
    print(f"Created {args.output.resolve()} ({args.output.stat().st_size} bytes, {len(set(files))} files)")
    print("Before submitting: verify Kaggle username, official measurements and verified score in README.")


if __name__ == "__main__":
    main()
