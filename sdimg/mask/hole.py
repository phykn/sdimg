import cv2
import numpy as np

from .helper import to_mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = to_mask(mask)
    if not np.any(mask):
        return mask

    inverted = (1 - mask) * 255
    filled = inverted.copy()
    flood_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(filled, flood_mask, (0, 0), 0)

    holes = filled > 0
    return (mask | holes).astype(np.uint8)
