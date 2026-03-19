import numpy as np


def pad_to_square(
    src: np.ndarray,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = src.shape[:2]
    bottom = max(0, width - height)
    right = max(0, height - width)
    top, left = 0, 0

    if src.ndim == 2:
        pad_width = ((top, bottom), (left, right))
    else:
        pad_width = ((top, bottom), (left, right), (0, 0))

    result = np.pad(src, pad_width, mode="constant", constant_values=0)
    meta = (top, bottom, left, right)
    if return_meta:
        return result, meta
    return result
