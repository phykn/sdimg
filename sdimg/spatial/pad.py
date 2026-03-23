import numpy as np


def pad_to_square(
    src: np.ndarray,
    return_box: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int, int]]:
    if not isinstance(src, np.ndarray):
        raise TypeError("src must be a numpy.ndarray.")
    if src.ndim not in {2, 3}:
        raise ValueError("src must have shape (H, W) or (H, W, C).")

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
