import numpy as np
from skimage.filters import threshold_otsu

from ..core import is_image, to_gray


def otsu_threshold(
    src: np.ndarray,
    threshold_offset: float = 0.0,
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    if threshold_offset < 0:
        raise ValueError("threshold_offset must be greater than or equal to 0.")

    gray = to_gray(src)
    threshold = float(threshold_otsu(gray))
    adjusted_threshold = max(0.0, float(threshold) - threshold_offset)
    return (gray > adjusted_threshold).astype(np.uint8)
