import numpy as np

from ..core import is_mask
from ._common import _finalize_image, _finalize_mask, _require_spatial_input


def crop(
    src: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    data = _require_spatial_input(src)
    wmin, hmin, wmax, hmax = _require_bbox(bbox, data.shape[:2])
    cropped = data[hmin:hmax, wmin:wmax]

    if is_mask(data):
        return _finalize_mask(cropped)

    return _finalize_image(cropped)


def _require_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError("bbox must be a 4-tuple of (wmin, hmin, wmax, hmax).")

    wmin, hmin, wmax, hmax = bbox
    values = (wmin, hmin, wmax, hmax)

    if any(not isinstance(value, int) for value in values):
        raise ValueError("bbox values must be integers.")

    height, width = shape

    if not (0 <= wmin < wmax <= width):
        raise ValueError("bbox width range is out of bounds or invalid.")

    if not (0 <= hmin < hmax <= height):
        raise ValueError("bbox height range is out of bounds or invalid.")

    return (wmin, hmin, wmax, hmax)
