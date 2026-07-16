import hashlib
import math

import numpy as np

from ..core.types import BBox
from ..core.validation import (
    validate_bbox,
    validate_finite,
    validate_positive_int,
    validate_source,
)
from .dtype import restore_dtype, validate_float64_integer_range


def split_tiles(
    array: np.ndarray,
    grid: int | tuple[int, int],
    overlap: float | tuple[float, float] = 0.0,
    return_metadata: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, object]]:
    array = validate_source(array)
    columns, rows = _parse_grid(grid)
    overlap_x, overlap_y = _parse_overlap(overlap)
    if not isinstance(return_metadata, bool):
        raise TypeError("return_metadata must be a bool.")

    starts_y, tile_height = _resolve_axis(array.shape[0], rows, overlap_y)
    starts_x, tile_width = _resolve_axis(array.shape[1], columns, overlap_x)
    tiles: list[np.ndarray] = []
    boxes: list[BBox] = []
    for ymin in starts_y:
        ymax = ymin + tile_height
        for xmin in starts_x:
            xmax = xmin + tile_width
            tiles.append(np.ascontiguousarray(array[ymin:ymax, xmin:xmax]).copy())
            boxes.append((xmin, ymin, xmax, ymax))

    if not return_metadata:
        return tiles
    return tiles, {"shape": array.shape, "boxes": boxes}


def merge_tiles(
    tiles: list[np.ndarray],
    metadata: dict[str, object],
) -> np.ndarray:
    if not isinstance(tiles, list):
        raise TypeError("tiles must be a list.")
    if not tiles:
        raise ValueError("tiles must not be empty.")
    shape, boxes = _validate_metadata(metadata)
    if len(tiles) != len(boxes):
        raise ValueError("tiles and metadata boxes length must match.")

    tiles = [validate_source(tile, name="tile") for tile in tiles]
    dtype = tiles[0].dtype
    coverage = np.zeros(shape[:2], dtype=np.int64)
    expected_ndim = len(shape)
    expected_channels = shape[2:] if expected_ndim == 3 else ()
    validated_boxes: list[BBox] = []

    for tile, bbox in zip(tiles, boxes):
        if tile.dtype != dtype:
            raise ValueError("all tiles must have the same dtype.")
        if tile.ndim != expected_ndim or tile.shape[2:] != expected_channels:
            raise ValueError("tile channel shape must match metadata shape.")
        validate_float64_integer_range(tile, "tile")
        xmin, ymin, xmax, ymax = validate_bbox(bbox, shape=shape[:2], name="tile bbox")
        if tile.shape[:2] != (ymax - ymin, xmax - xmin):
            raise ValueError("tile shape must match its metadata bbox.")
        validated_boxes.append((xmin, ymin, xmax, ymax))
        coverage[ymin:ymax, xmin:xmax] += 1

    if np.any(coverage == 0):
        raise ValueError("tile boxes must cover every output pixel.")
    if np.issubdtype(dtype, np.integer):
        return _merge_integer_tiles(tiles, validated_boxes, coverage, shape, dtype)
    return _merge_real_tiles(tiles, validated_boxes, coverage, shape, dtype)


def _merge_integer_tiles(
    tiles: list[np.ndarray],
    boxes: list[BBox],
    coverage: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    quotient_sum = np.zeros(shape, dtype=np.int64)
    remainder_sum = np.zeros(shape, dtype=np.int64)

    for tile, (xmin, ymin, xmax, ymax) in zip(tiles, boxes):
        counts = _broadcast_counts(
            coverage[ymin:ymax, xmin:xmax],
            tile.shape,
        )
        values = tile.astype(np.int64)
        quotient, remainder = np.divmod(values, counts)
        quotient_sum[ymin:ymax, xmin:xmax] += quotient
        remainder_sum[ymin:ymax, xmin:xmax] += remainder

    counts = _broadcast_counts(coverage, shape)
    rounded = quotient_sum + remainder_sum // counts
    remainder = remainder_sum % counts
    twice_remainder = 2 * remainder
    round_up = (twice_remainder > counts) | (
        (twice_remainder == counts) & (rounded % 2 != 0)
    )
    rounded += round_up
    return np.ascontiguousarray(rounded.astype(dtype))


def _merge_real_tiles(
    tiles: list[np.ndarray],
    boxes: list[BBox],
    coverage: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    ordered_tiles = sorted(
        zip(tiles, boxes),
        key=_tile_content_key,
    )
    scale = np.zeros(shape, dtype=np.float64)
    has_nan = np.zeros(shape, dtype=bool)
    has_positive_infinity = np.zeros(shape, dtype=bool)
    has_negative_infinity = np.zeros(shape, dtype=bool)
    for tile, (xmin, ymin, xmax, ymax) in ordered_tiles:
        values = tile.astype(np.float64)
        region = scale[ymin:ymax, xmin:xmax]
        finite_magnitudes = np.where(np.isfinite(values), np.abs(values), 0.0)
        np.maximum(region, finite_magnitudes, out=region)
        has_nan[ymin:ymax, xmin:xmax] |= np.isnan(values)
        has_positive_infinity[ymin:ymax, xmin:xmax] |= np.isposinf(values)
        has_negative_infinity[ymin:ymax, xmin:xmax] |= np.isneginf(values)

    counts = _broadcast_counts(coverage, shape)
    use_scaled = scale > np.finfo(np.float64).max / counts / 2.0
    total = np.zeros(shape, dtype=np.float64)
    compensation = np.zeros(shape, dtype=np.float64)

    for tile, (xmin, ymin, xmax, ymax) in ordered_tiles:
        values = tile.astype(np.float64)
        values = np.where(np.isfinite(values), values, 0.0)
        scaled = use_scaled[ymin:ymax, xmin:xmax]
        if np.any(scaled):
            values = values.copy()
            values[scaled] /= scale[ymin:ymax, xmin:xmax][scaled]
        _neumaier_add(
            total[ymin:ymax, xmin:xmax],
            compensation[ymin:ymax, xmin:xmax],
            values,
        )

    merged = (total + compensation) / counts
    merged[use_scaled] = (
        np.clip(merged[use_scaled], -1.0, 1.0) * scale[use_scaled]
    )
    conflicting_infinities = has_positive_infinity & has_negative_infinity
    invalid = has_nan | conflicting_infinities
    merged[has_positive_infinity & ~invalid] = np.inf
    merged[has_negative_infinity & ~invalid] = -np.inf
    merged[invalid] = np.nan
    return np.ascontiguousarray(restore_dtype(merged, dtype))


def _tile_content_key(item: tuple[np.ndarray, BBox]) -> tuple[object, ...]:
    tile, bbox = item
    contiguous = np.ascontiguousarray(tile)
    magnitudes = np.abs(contiguous.astype(np.float64, copy=False))
    maximum_magnitude = (
        np.inf if np.isnan(magnitudes).any() else float(magnitudes.max())
    )
    digest = hashlib.sha256(contiguous.view(np.uint8)).digest()
    return (maximum_magnitude, *bbox, tile.shape, digest)


def _neumaier_add(
    total: np.ndarray,
    compensation: np.ndarray,
    values: np.ndarray,
) -> None:
    updated = total + values
    total_larger = np.abs(total) >= np.abs(values)
    compensation += np.where(
        total_larger,
        (total - updated) + values,
        (values - updated) + total,
    )
    total[...] = updated


def _broadcast_counts(coverage: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 3:
        return np.broadcast_to(coverage[..., None], shape)
    return coverage


def _parse_grid(grid: object) -> tuple[int, int]:
    if isinstance(grid, int) and not isinstance(grid, bool):
        size = validate_positive_int(grid, "grid")
        return size, size
    if isinstance(grid, tuple) and len(grid) == 2:
        return (
            validate_positive_int(grid[0], "grid[0]"),
            validate_positive_int(grid[1], "grid[1]"),
        )
    raise TypeError("grid must be an int or a tuple of two ints.")


def _parse_overlap(overlap: object) -> tuple[float, float]:
    if isinstance(overlap, (int, float, np.integer, np.floating)) and not isinstance(
        overlap, bool
    ):
        value = _validate_overlap(overlap, "overlap")
        return value, value
    if isinstance(overlap, tuple) and len(overlap) == 2:
        return (
            _validate_overlap(overlap[0], "overlap[0]"),
            _validate_overlap(overlap[1], "overlap[1]"),
        )
    raise TypeError("overlap must be a number or a tuple of two numbers.")


def _validate_overlap(value: object, name: str) -> float:
    result = validate_finite(value, name)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= overlap < 1.")
    return result


def _resolve_axis(length: int, count: int, overlap: float) -> tuple[list[int], int]:
    if count == 1:
        return [0], length
    if count > length:
        raise ValueError("grid count cannot exceed the corresponding array length.")
    denominator = 1.0 + (count - 1) * (1.0 - overlap)
    tile_size = max(1, math.ceil(length / denominator))

    while tile_size <= length:
        span = length - tile_size
        starts = np.rint(np.linspace(0, span, num=count)).astype(np.int64).tolist()
        valid = all(
            right > left and 1.0 - ((right - left) / tile_size) + 1e-9 >= overlap
            for left, right in zip(starts, starts[1:])
        )
        if valid:
            return starts, tile_size
        tile_size += 1
    raise ValueError("unable to resolve tiles for the requested grid and overlap.")


def _validate_metadata(metadata: object) -> tuple[tuple[int, ...], list[BBox]]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict.")
    if set(metadata) != {"shape", "boxes"}:
        raise ValueError("metadata must contain exactly shape and boxes.")
    shape = metadata["shape"]
    boxes = metadata["boxes"]
    if not isinstance(shape, tuple) or len(shape) not in {2, 3}:
        raise ValueError("metadata['shape'] must be a tuple of length 2 or 3.")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in shape
    ):
        raise ValueError("metadata['shape'] values must be positive ints.")
    if not isinstance(boxes, list):
        raise ValueError("metadata['boxes'] must be a list.")
    return shape, boxes
