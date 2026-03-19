import numpy as np


def crop(
    src: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    wmin, hmin, wmax, hmax = bbox
    h, w = src.shape[:2]
    if wmin < 0 or wmax > w or hmin < 0 or hmax > h or wmin >= wmax or hmin >= hmax:
        raise ValueError("bbox is out of bounds or invalid.")
    return src[hmin:hmax, wmin:wmax]
