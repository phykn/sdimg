import math

import numpy as np

from ...core.types import BBox
from ...core.validation import (
    validate_finite,
    validate_positive_int,
    validate_source,
)


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
            tiles.append(array[ymin:ymax, xmin:xmax].copy(order="C"))
            boxes.append((xmin, ymin, xmax, ymax))

    if not return_metadata:
        return tiles
    return tiles, {"shape": array.shape, "boxes": boxes}


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
