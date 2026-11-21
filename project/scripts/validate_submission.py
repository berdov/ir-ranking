#!/usr/bin/env python3
"""Independent CSV integrity check against objects.csv and sample_submission.csv."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('data_dir',type=Path)
parser.add_argument('submission',type=Path)
parser.add_argument('--report_file',type=Path)
args=parser.parse_args()
with (args.data_dir/'objects.csv').open(newline='') as stream:
    objects=list(csv.DictReader(stream))
with (args.data_dir/'sample_submission.csv').open(newline='') as stream:
    sample=list(csv.DictReader(stream))
with args.submission.open(newline='') as stream:
    reader=csv.DictReader(stream)
    assert reader.fieldnames==['ObjectId','Score'], reader.fieldnames
    submission=list(reader)
ids=[row['ObjectId'] for row in submission]
expected=[row['ObjectId'] for row in objects]
assert ids==expected, 'ObjectId count/order differs from objects.csv'
assert len(set(ids))==len(ids), 'Duplicate ObjectId'
assert set(ids)=={row['ObjectId'] for row in sample}, 'Sample IDs differ'
scores=[float(row['Score']) for row in submission]
assert all(math.isfinite(value) for value in scores), 'Nonfinite score'
report={'rows':len(ids),'queries':len({row['QueryId'] for row in objects}),
        'ids_complete_unique_and_ordered':True,'all_scores_finite':True,
        'sha256':hashlib.sha256(args.submission.read_bytes()).hexdigest()}
if args.report_file:
    args.report_file.parent.mkdir(parents=True,exist_ok=True)
    args.report_file.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
