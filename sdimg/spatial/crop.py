import numpy as np


def crop(
    src: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    if not isinstance(src, np.ndarray):
        raise TypeError("src must be a numpy.ndarray.")
    if src.ndim not in {2, 3}:
        raise ValueError("src must have shape (H, W) or (H, W, C).")

    wmin, hmin, wmax, hmax = bbox
    h, w = src.shape[:2]
    if wmin < 0 or wmax > w or hmin < 0 or hmax > h or wmin >= wmax or hmin >= hmax:
        raise ValueError("bbox is out of bounds or invalid.")
    return src[hmin:hmax, wmin:wmax]
