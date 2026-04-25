import numpy as np

from .._core.types import BBox
from .._core.validate import ensure_src


def pad_to_square(
    src: np.ndarray,
    return_box: bool = False,
) -> np.ndarray | tuple[np.ndarray, BBox]:
    src = ensure_src(src, name="src")

    height, width = src.shape[:2]
    bottom = max(0, width - height)
    right = max(0, height - width)
    top, left = 0, 0

    if src.ndim == 2:
        pad_width = ((top, bottom), (left, right))
    else:
        pad_width = ((top, bottom), (left, right), (0, 0))

    result = np.pad(src, pad_width, mode="constant", constant_values=0)
    box = (left, top, left + width, top + height)
    if return_box:
        return result, box
    return result
