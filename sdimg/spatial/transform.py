import numpy as np

from ..core import is_mask
from ._common import _finalize_image, _require_spatial_input


def rotate(
    src: np.ndarray,
    rotation: int = 0,
) -> np.ndarray:
    data = _require_spatial_input(src)
    k = _resolve_rotation_count(rotation)
    rotated = np.rot90(data, k=k)

    if is_mask(data):
        return rotated.astype(np.uint8, copy=False)

    return _finalize_image(rotated)


def flip(
    src: np.ndarray,
    direction: str,
) -> np.ndarray:
    data = _require_spatial_input(src)

    if direction == "horizontal":
        flipped = np.flip(data, axis=1)
    elif direction == "vertical":
        flipped = np.flip(data, axis=0)
    elif direction == "transpose":
        flipped = np.swapaxes(data, 0, 1)
    else:
        raise ValueError("direction must be one of horizontal, vertical, transpose.")

    if is_mask(data):
        return flipped.astype(np.uint8, copy=False)

    return _finalize_image(flipped)


def _resolve_rotation_count(rotation: int) -> int:
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("rotation must be one of 0, 90, 180, 270.")

    return rotation // 90
