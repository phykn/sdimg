import cv2
import numpy as np
from typing import Literal

from .helper import to_mask
from .pad import _pad_1px, _unpad_1px

DistanceType = Literal["l1", "l2", "c"]


def distance_transform(
    mask: np.ndarray,
    distance_type: DistanceType = "l2",
    mask_size: int = 3,
) -> np.ndarray:
    mask = to_mask(mask)

    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.float32)

    if distance_type == "l1":
        cv2_distance_type = cv2.DIST_L1
    elif distance_type == "l2":
        cv2_distance_type = cv2.DIST_L2
    elif distance_type == "c":
        cv2_distance_type = cv2.DIST_C
    else:
        raise ValueError("distance_type must be one of: 'l1', 'l2', 'c'.")

    padded = _pad_1px(mask)
    distance = cv2.distanceTransform(
        padded,
        cv2_distance_type,
        mask_size,
        dstType=cv2.CV_32F,
    )
    distance = _unpad_1px(distance)
    return distance
