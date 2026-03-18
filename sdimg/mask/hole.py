import cv2
import numpy as np

from ..core import to_mask


def fill_holes(src: np.ndarray) -> np.ndarray:
    mask = to_mask(src)

    if np.count_nonzero(mask) == 0:
        return mask

    background = (1 - mask) * 255
    filled = background.copy()
    flood_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(
        filled,
        flood_mask,
        (0, 0),
        0,
    )

    holes = (filled > 0).astype(np.uint8)
    return ((mask > 0) | (holes > 0)).astype(np.uint8)
