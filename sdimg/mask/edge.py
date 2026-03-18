import cv2
import numpy as np

from ..core import to_mask


def extract_edge(
    src: np.ndarray,
    ksize: tuple[int, int] = (3, 3),
) -> np.ndarray:
    mask = to_mask(src)

    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    eroded = cv2.erode(mask, kernel)
    edge = mask - eroded
    return (edge > 0).astype(np.uint8)
