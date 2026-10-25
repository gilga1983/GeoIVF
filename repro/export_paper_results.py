#!/usr/bin/env python3
"""Regenerate NavHints plot data from bundled aggregates, without a private repo.

  python3 repro/export_paper_results.py --output artifacts/paper
  cd artifacts/paper && pdflatex -interaction=nonstopmode -halt-on-error figures.tex

Python standard library only. No input dataset downloads, ANN builds or SSD
measurements. Original plotting/derivation sources are in repro/paper/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import zipfile
from pathlib import Path

import verify_review_bundle as check

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'repro/paper'


def module(name):
    spec = importlib.util.spec_from_file_location(name, PAPER / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def emit(directory, name, fields, rows):
    with (directory / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compare_reference(directory):
    """Compare regenerated values with the submitted paper, including NaN gaps."""
    count = 0
    for ref in sorted((PAPER / 'reference-data').glob('*.csv')):
        with ref.open(newline='') as f:
            expected = list(csv.DictReader(f))
        with (directory / ref.name).open(newline='') as f:
            actual = list(csv.DictReader(f))
        check.require(len(actual) == len(expected), ref.name + ': wrong row count')
        for new, old in zip(actual, expected):
            check.require(new.keys() == old.keys(), ref.name + ': wrong columns')
            for col in old:
                try:
                    a, b = float(new[col]), float(old[col])
                except ValueError:
                    check.require(new[col] == old[col], f'{ref.name}:{col}: text mismatch')
                else:
                    check.require((math.isnan(a) and math.isnan(b)) or
                                  math.isclose(a, b, rel_tol=0, abs_tol=.00011),
                                  f'{ref.name}:{col}: {a} differs from manuscript {b}')
        count += 1
    print(f'  Paper CSVs  {count} regenerated files match the submitted reference values  PASS')
    return count


def export(z, out):
    def read(name): return json.loads(z.read('evidence/' + name))
    data = out / 'figures/data'
    data.mkdir(parents=True, exist_ok=True)
    front = module('derive_final_core_eval')
    service = module('derive_service_latency')
    components = []
    for idx, key in enumerate(check.DATASETS):
        raw = read(key + '/' + check.DATASETS[key][1])
        curves = {m: front.frontier(raw['summary'][m]) for m in ('baseline', 'ivf', 'core')}
        matched = []
        for L, point in sorted(raw['summary']['core'].items(), key=lambda x: int(x[0])):
            recall = float(point['recall_percent'])
            at = {m: front.interpolate(c, recall) for m, c in curves.items()}
            if any(v is None for v in at.values()): continue
            B, J, C = [at[m]['reads'] for m in ('baseline', 'ivf', 'core')]
            matched.append({'recall': recall, 'junction':100*(1-J/B), 'navhints':100*(1-C/B)})
        components.append({'dataset_id':idx, 'dataset':('Coveo','PubMed','Text2Image-10M','BigANN-10M')[idx],
            **{col:round(statistics.mean(p[col] for p in matched),4) for col in ('junction','navhints')},
            'anchors':len(matched), 'recall_min':round(min(p['recall'] for p in matched),4),
            'recall_max':round(max(p['recall'] for p in matched),4)})
        for method, curve in curves.items():
            name = 'junction' if method == 'ivf' else method
            emit(data, f'eval_frontier_{key}_{name}.csv', ['recall','ssd_reads','latency_us'],
                 [{'recall':round(p['recall'],6), 'ssd_reads':round(p['reads'],6),
                   'latency_us':round(p['latency'],4)} for p in curve])
    emit(data,'eval_components.csv',list(components[0]),components)

    latency = []
    for key, dataset, filename, label in service.DATASETS:
        raw = read('service-' + key + '/' + check.SERVICE[key][0])
        result = service.calculate(raw, dataset)
        service.write_csv(data / filename, result)
        result.pop('plot')
        latency.append({'label':label,'result':result})
    (out/'service-latency.json').write_text(json.dumps(latency,indent=2)+'\n')

    warm = read('figure8-warmup-k1-k2-k4-k8/final-core-warmup-starts.json')
    rows = [{'history':row['history_queries'],
             **{f'k{k}':row['starts'][str(k)]['mean_io_saving_percent'] for k in check.STARTS}}
            for row in warm['windows']]
    emit(data,'eval_warmup_starts_final_core.csv',list(rows[0]),rows)
    locality = read('figure8-locality-k1-k2-k4-k8/final-core-starts.json')
    rows = [{'repeats':p['requests_per_identity'],
             **{f'k{k}':p['starts'][str(k)]['vs_junctions']['io_saving_percent'] for k in check.STARTS}}
            for p in locality['pools']]
    emit(data,'eval_locality_starts_final_core.csv',list(rows[0]),rows)

    overlap = {}
    for key, source in (('pubmed','pubmed1M'),('coveo','coveo'),
                        ('text2image','text2image-10M'),('bigann','bigann-10M')):
        raw = read('motivation-destination-reuse/' + source + '.json')
        overlap[key] = {r['capacity']:r['any_exact_top10_present_pct'] for r in raw['results']}
        junction = read('motivation-junction-concentration/' + source + '.json')
        emit(data,'motivation_junctions_' + key + '.csv',
             ['corpus_percent','heldout_visits_percent','unseen_ids'],
             [{'corpus_percent':100*r['corpus_fraction'],
               'heldout_visits_percent':100*r['heldout_visit_share'],
               'unseen_ids':r['ids_not_seen_in_training']} for r in junction['curve']])
    emit(data,'recent_destination_overlap.csv',['capacity',*overlap],
         [{'capacity':c, **{key:vals[c] for key,vals in overlap.items()}}
          for c in (0,16,32,64,128,256,512,1024,2048)])
    table5 = check.destination_controls(z)
    emit(data,'table5_destination_controls.csv',list(table5[0]),table5)
    scoring = []
    for key in check.DATASETS:
        raw = read('junction-score-' + key + '/junction-score-ablation.json')
        scoring.append({'dataset':key,'junctions':raw['controller']['B'],
                        'targets':len(raw['matched']),
                        'vs_visits_pct':statistics.mean(r['sum_vs_count_saving_percent'] for r in raw['matched']),
                        'vs_mean_position_pct':statistics.mean(r['sum_vs_mean_saving_percent'] for r in raw['matched'])})
    emit(data,'table7_junction_scoring.csv',list(scoring[0]),scoring)
    return compare_reference(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'repro/navhints-reviewer-evidence.zip')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/paper')
    p.add_argument('--overwrite', action='store_true', help='Overwrite generated files in an existing output directory')
    a = p.parse_args()
    release = json.loads((ROOT/'repro/release-20261025.json').read_text())
    check.require(hashlib.sha256(a.bundle.read_bytes()).hexdigest() == release['sha256'],
                  'Evidence ZIP differs from the frozen public release checksum')
    # Check every pinned reference/plot source before using it.
    assets = json.loads((PAPER/'SOURCE.json').read_text())
    for item in assets['files']:
        check.require(hashlib.sha256((PAPER/item['path']).read_bytes()).hexdigest() == item['sha256'],
                      'Paper source snapshot changed: ' + item['path'])
    summary = json.loads((ROOT/'repro/current-paper-summary-20261017.json').read_text())
    with zipfile.ZipFile(a.bundle) as z:
        check.archive_checks(z,check.check_source_manifests())
        check.primary(z,summary)
        check.latency(z,summary)
        check.newest_ablations(z,summary)
        check.destination_controls(z)
        check.require(not a.output.exists() or not any(a.output.iterdir()) or a.overwrite,
                      'Output directory is nonempty; choose a new path or pass --overwrite')
        a.output.mkdir(parents=True,exist_ok=True)
        # Static manuscript plot/table sources are copied exactly. Numeric plot
        # CSVs are independently regenerated below, not copied from references.
        for directory in ('figures','tables','scripts'):
            shutil.copytree(PAPER/directory,a.output/directory,dirs_exist_ok=True)
        for name in ('macros.tex','figures.tex','SOURCE.json'):
            shutil.copyfile(PAPER/name,a.output/name)
        count = export(z,a.output)
    (a.output/'REPRODUCTION.json').write_text(json.dumps({
        'evidence_sha256':release['sha256'],'paper_commit':assets['paper_commit'],
        'regenerated_reference_csvs':count,'ssd_measurements_rerun':False,
        'tables':'Pinned manuscript TeX tables; numerical verification coverage is listed in ARTIFACT_MAP.md',
    },indent=2)+'\n')
    print('PASS: standalone paper plotting directory prepared:',a.output)
    print('Optional figure/table rendering: cd',a.output,'&& pdflatex figures.tex && pdflatex figures.tex')


if __name__ == '__main__':
    try: main()
    except (ValueError,OSError,KeyError,zipfile.BadZipFile) as e:
        raise SystemExit('FAIL: ' + str(e))
