import cv2
import numpy as np

from .pad import pad_1px, unpad_1px


def extract_edge(
    mask: np.ndarray,
    ksize: tuple[int, int] = (3, 3),
) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    padded = pad_1px(mask)
    edge = padded - cv2.erode(padded, kernel)
    edge = unpad_1px(edge)
    return (edge > 0).astype(np.uint8)
