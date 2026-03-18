import cv2
import numpy as np

from ..core import to_mask


def keep_largest_component(
    src: np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    mask = to_mask(src)

    if np.count_nonzero(mask) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=connectivity,
    )

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = int(np.argmax(areas)) + 1
    return (labels == largest_index).astype(np.uint8)
