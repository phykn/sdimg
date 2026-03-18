import cv2
import numpy as np

from ..core import to_mask


DISTANCE_TYPES = {
    "l1": cv2.DIST_L1,
    "l2": cv2.DIST_L2,
    "c": cv2.DIST_C,
}


def distance_transform(
    src: np.ndarray,
    distance_type: str = "l2",
    mask_size: int = 3,
) -> np.ndarray:
    mask = to_mask(src)

    if np.count_nonzero(mask) == 0:
        return np.zeros(mask.shape, dtype=np.float32)

    if distance_type not in DISTANCE_TYPES:
        raise ValueError("distance_type must be one of: 'l1', 'l2', 'c'.")

    distance = cv2.distanceTransform(
        mask,
        DISTANCE_TYPES[distance_type],
        mask_size,
        dstType=cv2.CV_32F,
    )
    return distance.astype(np.float32, copy=False)
