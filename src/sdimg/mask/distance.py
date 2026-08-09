from typing import Literal

import cv2
import numpy as np

from .conversion import convert_to_mask

DistanceType = Literal["l1", "l2", "c"]


def compute_distance_transform(
    mask: np.ndarray,
    distance_type: DistanceType = "l2",
    mask_size: int = 3,
) -> np.ndarray:
    binary = convert_to_mask(mask)
    mapping = {"l1": cv2.DIST_L1, "l2": cv2.DIST_L2, "c": cv2.DIST_C}
    if not isinstance(distance_type, str):
        raise TypeError("distance_type must be a str.")
    if distance_type not in mapping:
        raise ValueError("distance_type must be one of: 'l1', 'l2', 'c'.")
    if not isinstance(mask_size, int) or isinstance(mask_size, bool):
        raise TypeError("mask_size must be an int.")
    allowed = {3, 5} if distance_type == "l2" else {3}
    if mask_size not in allowed:
        raise ValueError(
            f"mask_size must be one of {sorted(allowed)} for {distance_type}."
        )
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.float32)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    try:
        distance = cv2.distanceTransform(
            padded,
            mapping[distance_type],
            mask_size,
            dstType=cv2.CV_32F,
        )
    except Exception as exc:
        raise RuntimeError(f"compute_distance_transform failed: {exc}") from exc
    return distance[1:-1, 1:-1].astype(np.float32, copy=False)
