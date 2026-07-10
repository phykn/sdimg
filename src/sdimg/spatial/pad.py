import numpy as np

from ..core.types import BBox
from ..core.validation import validate_source


def pad_to_square(
    array: np.ndarray,
    return_bbox: bool = False,
) -> np.ndarray | tuple[np.ndarray, BBox]:
    array = validate_source(array)
    if not isinstance(return_bbox, bool):
        raise TypeError("return_bbox must be a bool.")

    height, width = array.shape[:2]
    bottom = max(0, width - height)
    right = max(0, height - width)
    pad_width = ((0, bottom), (0, right))
    if array.ndim == 3:
        pad_width += ((0, 0),)
    result = np.pad(array, pad_width, mode="constant", constant_values=0)
    result = np.ascontiguousarray(result)
    bbox = (0, 0, width, height)
    return (result, bbox) if return_bbox else result
