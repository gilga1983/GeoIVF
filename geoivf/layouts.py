"""Freeze physical pages, then rebuild RAM descriptions without moving payloads.

The legacy builder is retained for compatibility. Qualification uses a fixed
packing dimension and reuses the resulting page file for every RAM budget.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from .index import build_assigned, Index


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def freeze_layout(x, centers, labels, out, *, layout='geopack', pack_dims=16,
                  page_size=4096, provenance=None):
    """Build one immutable layout; summary precision is not a layout input."""
    out = Path(out)
    meta = build_assigned(x, centers, labels, out, dims=pack_dims, balls=1,
                          layout=layout, page_size=page_size, provenance=provenance)
    # Full coordinate ordering supports later summaries larger than pack_dims.
    sums = np.zeros(x.shape[1]); squares = np.zeros_like(sums)
    for s in range(0, len(x), 32768):
        a = x[s:s+32768].astype(np.float64)
        sums += a.sum(axis=0); squares += np.einsum('ij,ij->j', a, a)
    var = np.maximum(0, squares/len(x)-(sums/len(x))**2)
    ranking = np.argsort(-var, kind='stable').astype(np.int32)
    low = x.min(axis=0).astype(np.float64)
    scale = (x.max(axis=0).astype(np.float64)-low)/255
    scale[scale == 0] = 1
    np.savez(out/'summary_basis.npz', ranking=ranking, origin=low, scale=scale)
    meta.update(layout_payload_sha256=sha256(out/'vectors.pages'),
                layout_directory_sha256=sha256(out/'directory.npz'),
                pack_dims=pack_dims, physical_layout_frozen=True)
    (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


def summarize(layout_dir, out, *, dims=16, balls=8):
    """Create a format-v2 index sharing the exact existing FP32 page payload.

    The vectorized construction bounds quantization error against original points,
    not against pre-quantized representatives. The page IDs/order are never changed.
    """
    source, out = Path(layout_dir).resolve(), Path(out)
    base = Index(source); m = base.meta
    cap, d, npages = m['capacity'], m['d'], m['n_pages']
    if not m.get('physical_layout_frozen'):
        raise ValueError('expected a frozen layout')
    if not 1 <= dims <= d or not 1 <= balls <= cap:
        raise ValueError('invalid summary dimensions or ball count')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source/'summary_basis.npz', allow_pickle=False) as b:
        coordinates = b['ranking'][:dims]
        origin, scale = b['origin'][coordinates], b['scale'][coordinates]
    with np.load(source/'directory.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files
                  if k not in ('codes', 'radii', 'coordinates', 'origin', 'scale')}
    codes = np.zeros((npages, balls, dims), dtype=np.uint8)
    radii = np.full((npages, balls), -1, dtype=np.float32)
    raw = np.memmap(source/'vectors.pages', dtype='<f4', mode='r',
                    shape=(npages, m['page_size']//4))
    groups = np.array_split(np.arange(cap), balls)
    for start in range(0, npages, 2048):
        stop = min(npages, start+2048)
        points = np.asarray(raw[start:stop, :cap*d]).reshape(-1, cap, d)
        z = points[:, :, coordinates].astype(np.float64)
        valid = np.arange(cap)[None, :] < base.valid[start:stop, None]
        for j, slots in enumerate(groups):
            good = valid[:, slots]; count = good.sum(axis=1)
            a = z[:, slots, :]
            center = (a*good[:, :, None]).sum(axis=1)/np.maximum(count, 1)[:, None]
            code = np.clip(np.rint((center-origin)/scale), 0, 255).astype(np.uint8)
            decoded = origin + code*scale
            dist = np.linalg.norm(a-decoded[:, None, :], axis=2)
            radius = np.where(good, dist, 0).max(axis=1)
            guard = 128*np.finfo(float).eps*d*(1+np.abs(a).max(axis=(1, 2))
                                               +np.abs(decoded).max(axis=1))
            rr = np.nextafter((radius+guard).astype(np.float32), np.float32(np.inf))
            rr[count == 0] = -1
            codes[start:stop, j] = code
            radii[start:stop, j] = rr
    del raw
    arrays.update(coordinates=coordinates, origin=origin, scale=scale,
                  codes=codes, radii=radii)
    np.savez(out/'directory.npz', **arrays)
    try:
        os.link(source/'vectors.pages', out/'vectors.pages')
        sharing = 'hardlink'
    except OSError:
        shutil.copyfile(source/'vectors.pages', out/'vectors.pages')
        sharing = 'byte-identical-copy'
    if sha256(out/'vectors.pages') != m['layout_payload_sha256']:
        raise ValueError('payload hash changed while building summary')
    meta = dict(m, dims=dims, balls=balls, summary_bytes=codes.nbytes+radii.nbytes,
                directory_array_bytes=sum(a.nbytes for a in arrays.values()),
                summary_source_layout=str(source), payload_sharing=sharing)
    (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta
