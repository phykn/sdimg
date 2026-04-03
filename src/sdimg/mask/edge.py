import cv2
import numpy as np

from .helper import to_mask
from .pad import _pad_1px, _unpad_1px


def extract_edge(
    mask: np.ndarray,
    ksize: tuple[int, int] = (3, 3),
) -> np.ndarray:
    mask = to_mask(mask)

    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    padded = _pad_1px(mask)
    edge = padded - cv2.erode(padded, kernel)
    edge = _unpad_1px(edge)
    return (edge > 0).astype(np.uint8)
