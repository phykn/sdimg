import cv2
import numpy as np

from .convert import to_mask


def extract_edge(
    mask: np.ndarray,
    ksize: tuple[int, int] = (3, 3),
) -> np.ndarray:
    mask = to_mask(mask)

    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    edge = padded - cv2.erode(padded, kernel)
    edge = edge[1:-1, 1:-1]
    return (edge > 0).astype(np.uint8)
