import cv2
import numpy as np

from .conversion import convert_to_mask


def keep_largest_component(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    binary = convert_to_mask(mask)
    if type(connectivity) is not int:
        raise TypeError("connectivity must be an int.")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8.")
    if not np.any(binary):
        return binary
    try:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=connectivity,
        )
    except Exception as exc:
        raise RuntimeError(f"keep_largest_component failed: {exc}") from exc
    if count <= 1:
        return binary
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == largest).astype(np.uint8)
