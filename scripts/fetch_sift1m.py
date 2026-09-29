#!/usr/bin/env python3
"""Cache the canonical TexMex archive once. Never substitute another dataset."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import numpy as np

SOURCES = [
    'https://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz',
    'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz',
]
EXPECTED = {'sift_base.fvecs': (1000000, 128), 'sift_query.fvecs': (10000, 128),
            'sift_groundtruth.ivecs': (10000, 100), 'sift_learn.fvecs': (100000, 128)}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(root):
    result = {}
    for name, (n, d) in EXPECTED.items():
        path = root/name
        if path.stat().st_size != n*(d+1)*4:
            raise ValueError(f'incorrect canonical file size: {name}')
        a = np.memmap(path, dtype='<i4', mode='r', shape=(n, d+1))
        if not np.all(a[:, 0] == d):
            raise ValueError(f'inconsistent row dimensions: {name}')
        if name.endswith('.ivecs'):
            if a[:, 1:].min() < 0 or a[:, 1:].max() >= 1000000:
                raise ValueError('ground truth ID out of bounds')
        else:
            for s in range(0, n, 32768):
                if not np.isfinite(a[s:s+32768, 1:].view('<f4')).all():
                    raise ValueError('nonfinite dataset coordinates')
        result[name] = dict(rows=n, dimensions=d, bytes=path.stat().st_size,
                           sha256=digest(path))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=Path.home()/'.cache/geoivf/datasets')
    p.add_argument('--report', type=Path)
    args = p.parse_args(); args.cache.mkdir(parents=True, exist_ok=True)
    target = args.cache/'sift1m'
    with open(args.cache/'sift1m.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target/'dataset.json').exists():
            report = json.loads((target/'dataset.json').read_text())
            if validate(target) != report['files']:
                raise ValueError('cached dataset changed; refusing silent reuse')
        else:
            if target.exists():
                raise FileExistsError(f'incomplete cache requires inspection: {target}')
            with tempfile.TemporaryDirectory(prefix='sift1m-', dir=args.cache) as t:
                t = Path(t); archive = t/'sift.tar.gz'; source = None
                for url in SOURCES:
                    print(f'Downloading {url}', flush=True)
                    rc = subprocess.run(['curl', '-fL', '--connect-timeout', '15',
                         '--max-time', '600', '--retry', '1', '--output', str(archive), url])
                    if rc.returncode == 0:
                        source = url; break
                if source is None:
                    raise RuntimeError('canonical SIFT1M download failed; no substitute used')
                extracted = t/'files'; extracted.mkdir(); found = set()
                with tarfile.open(archive, 'r:gz') as tar:
                    for entry in tar:
                        name = Path(entry.name).name
                        if name not in EXPECTED:
                            continue
                        if name in found or not entry.isfile():
                            raise ValueError('duplicate or nonregular canonical member')
                        n, d = EXPECTED[name]
                        if entry.size != n*(d+1)*4:
                            raise ValueError('unexpected canonical archive member size')
                        with tar.extractfile(entry) as src, (extracted/name).open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                        found.add(name)
                if found != set(EXPECTED):
                    raise ValueError('canonical archive is missing required members')
                report = dict(dataset='TexMex ANN_SIFT1M (not a SIFT1B subset)',
                              source=source, archive_sha256=digest(archive),
                              downloaded_unix=time.time(), files=validate(extracted),
                              checksum_status='observed hashes, not upstream-signed checksums')
                (extracted/'dataset.json').write_text(json.dumps(report, indent=2)+'\n')
                os.rename(extracted, target)
        print(json.dumps(report, indent=2), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
