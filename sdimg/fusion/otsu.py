import numpy as np
from skimage.filters import threshold_otsu

from ..image.helper import to_gray


def otsu_threshold(
    image: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    if scale < 0:
        raise ValueError("scale must be greater than or equal to 0.")

    gray = to_gray(image)
    threshold = float(threshold_otsu(gray))
    adjusted_threshold = threshold * scale
    return (gray > adjusted_threshold).astype(np.uint8)
