#!/usr/bin/env python3
"""Build the selected-NavHints starts/locality plot from frozen real-SSD evidence.

Usage:
  python3 scripts/derive_starts_locality.py --artifact path/to/run.zip \
      --existing figures/data/eval_locality_final_core.csv \
      --output figures/data/eval_locality_starts_final_core.csv

Fail closed if controller, six recall anchors, original k=8 curve, or
no-persistence checks disagree with the paper's frozen configuration.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path

COMMIT = '59e0e63eedde4942a1f2b5794dc581df9c80141b'
POOLS = (5000, 2000, 1000, 500, 100)
STARTS = (1, 2, 4, 8)
WIDTHS = (10, 20, 40, 80, 160, 320)
TARGETS = (35., 45., 55., 65., 72., 75.)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', required=True, type=Path)
    ap.add_argument('--existing', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if not args.artifact.is_file() or not args.existing.is_file():
        raise FileNotFoundError('Frozen ZIP and existing k=8 curve are required')
    with zipfile.ZipFile(args.artifact) as z:
        if z.read('geoivf-commit.txt').decode().strip() != COMMIT:
            raise ValueError('Source SHA differs from frozen experiment')
        data = json.loads(z.read('starts/final-core-starts.json'))
        if data.get('core') != {
            'junction_budget': 14464, 'recent_history': 2048,
            'regions': 512, 'starts_tested': [0, 1, 2, 4, 8],
            'persistent_exits': False,
        }:
            raise ValueError('Non-final controller or persistent exits enabled')
        if tuple(map(float, data['targets'])) != TARGETS:
            raise ValueError('Recall-target grid differs from frozen study')
        pools = data['pools']
        if [int(p['working_set_queries']) for p in pools] != list(POOLS):
            raise ValueError('Unexpected workload identity counts/order')
        rows = []
        for pool in pools:
            population = int(pool['working_set_queries'])
            repetitions = 5000 / population
            if not math.isclose(pool['requests_per_identity'], repetitions):
                raise ValueError('Bad workload repetition count')
            if int(pool['zero_starts']['compared_to_junction']['points']) != 6:
                raise ValueError('Junction-only reference does not support six recall anchors')
            row = {'repeats': f'{repetitions:.1f}'}
            if set(pool['starts']) != set(map(str, STARTS)):
                raise ValueError('Missing or extra starts ablation')
            for k in STARTS:
                arm = pool['starts'][str(k)]
                summary = arm['vs_junctions']
                if summary['points'] != 6:
                    raise ValueError(f'{population} identities, k={k}: missing recall target')
                if sorted(int(r['search_l']) for r in arm['per_L']) != list(WIDTHS):
                    raise ValueError(f'{population} identities, k={k}: incomplete L grid')
                v = float(summary['io_saving_percent'])
                if not math.isfinite(v) or not -100 <= v <= 100:
                    raise ValueError('Invalid relative read saving')
                # Every k arm uses RAM-only Core with NO graph-page writes.
                log = z.read(f'starts/pool{population}-starts{k}.log').decode()
                matches = re.findall(
                    r'EXPERIENCE_STATS L=(\d+) cache_capacity=(\d+) hub_capacity=(\d+).*? eval_writes=(\d+)',
                    log)
                if sorted((int(a), int(b), int(c), int(d))
                          for a, b, c, d in matches) != [
                              (L, 2048, 0, 0) for L in WIDTHS
                          ]:
                    raise ValueError(f'{population} identities, k={k}: unverified RAM-only replay')
                row[f'k{k}'] = f'{v:.6f}'
            rows.append(row)

    with args.existing.open(newline='') as f:
        existing = list(csv.DictReader(f))
    if len(existing) != len(rows):
        raise ValueError('Number of original k=8 locality points changed')
    for new, old in zip(rows, existing):
        if not math.isclose(float(new['repeats']), float(old['repeats']), abs_tol=1e-9) or not math.isclose(float(new['k8']), float(old['core']), abs_tol=1e-5):
            raise ValueError(f'New k=8 data disagree with published locality panel: {new} vs {old}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['repeats', 'k1', 'k2', 'k4', 'k8'],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'NAVHINTS_FINAL_CORE_STARTS_LOCALITY_VALIDATED {args.output}')
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
