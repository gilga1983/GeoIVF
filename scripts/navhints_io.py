"""Read and slice the binary inputs used by NavHints experiment replays.

Vector files start with two little-endian uint32 values (rows, dimensions),
followed by a row-major payload. Ground-truth files use the same header for
(queries, neighbors), followed by one uint32 ID block and, optionally, a
float32 distance block. Distances are a separate block, not interleaved IDs.

These helpers only prepare replay inputs. They never perform ANN search.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import struct
import tempfile
from typing import BinaryIO, Iterator

HEADER = struct.Struct("<II")
VECTOR_COPY_CHUNK_BYTES = 16 << 20
GROUND_TRUTH_COPY_CHUNK_BYTES = 8 << 20


def read_xbin_shape(path: Path) -> tuple[int, int]:
    """Read a nonempty vector/ground-truth shape without loading its payload."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError(f"{path}: expected an 8-byte rows/dimensions header")
    rows, dimensions = HEADER.unpack(raw)
    if rows == 0 or dimensions == 0:
        raise ValueError(f"{path}: invalid empty shape ({rows}, {dimensions})")
    return rows, dimensions


def _validate_slice(
    source: Path, destination: Path, rows: int, start: int, count: int
) -> None:
    if start < 0 or count <= 0 or start + count > rows:
        raise ValueError(
            f"{source}: requested rows [{start}, {start + count}) "
            f"outside the {rows}-row input; count must be positive"
        )
    if source.resolve() == destination.resolve() or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError(f"{source}: slice output must differ from its source")


@contextmanager
def _atomic_output(destination: Path) -> Iterator[BinaryIO]:
    """Replace the destination only after its complete payload has been copied."""
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", dir=destination.parent, delete=False
    )
    staged_path = Path(temporary.name)
    try:
        with temporary as output:
            yield output
        staged_path.replace(destination)
    finally:
        staged_path.unlink(missing_ok=True)


def _copy_bytes(
    source: BinaryIO,
    destination: BinaryIO,
    count: int,
    chunk_bytes: int,
    label: str,
) -> None:
    remaining = count
    while remaining:
        block = source.read(min(chunk_bytes, remaining))
        if not block:
            raise ValueError(f"{label}: truncated payload; {remaining} bytes missing")
        destination.write(block)
        remaining -= len(block)


def slice_vectors(
    source: Path, destination: Path, start: int, count: int, itemsize: int = 4
) -> None:
    """Copy contiguous vectors while preserving their original element bytes.

    Use itemsize=4 for float32 fbin inputs and itemsize=1 for uint8 bbin inputs.
    A failed copy leaves an existing destination intact.
    """
    source, destination = Path(source), Path(destination)
    rows, dimensions = read_xbin_shape(source)
    _validate_slice(source, destination, rows, start, count)
    if itemsize <= 0:
        raise ValueError(f"{source}: element size must be positive, got {itemsize}")
    row_bytes = dimensions * itemsize
    with source.open("rb") as input_stream, _atomic_output(destination) as output:
        input_stream.seek(HEADER.size + start * row_bytes)
        output.write(HEADER.pack(count, dimensions))
        _copy_bytes(
            input_stream,
            output,
            count * row_bytes,
            VECTOR_COPY_CHUNK_BYTES,
            str(source),
        )


def slice_ground_truth(
    source: Path,
    destination: Path,
    start: int,
    count: int,
    *,
    require_distances: bool = False,
) -> None:
    """Copy query rows from ID-only or ID-plus-distance ground-truth files.

    Coveo's frozen format requires both blocks. Other public benchmarks may
    supply IDs alone; callers can retain their own format contract explicitly.
    """
    source, destination = Path(source), Path(destination)
    rows, neighbors = read_xbin_shape(source)
    _validate_slice(source, destination, rows, start, count)
    row_bytes = neighbors * 4
    id_block_bytes = rows * row_bytes
    size_bytes = source.stat().st_size
    ids_only_size = HEADER.size + id_block_bytes
    with_distances_size = ids_only_size + id_block_bytes
    if size_bytes not in (ids_only_size, with_distances_size):
        raise ValueError(
            f"{source}: unsupported ground-truth size {size_bytes}; "
            f"expected {ids_only_size} ID-only bytes or {with_distances_size} bytes with distances"
        )
    has_distances = size_bytes == with_distances_size
    if require_distances and not has_distances:
        raise ValueError(
            f"{source}: this replay requires a ground-truth distance block"
        )
    block_offsets = [HEADER.size]
    if has_distances:
        block_offsets.append(HEADER.size + id_block_bytes)
    with source.open("rb") as input_stream, _atomic_output(destination) as output:
        output.write(HEADER.pack(count, neighbors))
        for offset in block_offsets:
            input_stream.seek(offset + start * row_bytes)
            _copy_bytes(
                input_stream,
                output,
                count * row_bytes,
                GROUND_TRUTH_COPY_CHUNK_BYTES,
                str(source),
            )
