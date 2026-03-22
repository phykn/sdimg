import numpy as np
from skimage.filters import threshold_otsu

from ..image.helper import to_gray


def otsu_threshold(
    image: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Binarize an image using Otsu's threshold.

    Args:
        scale: Multiplier for the computed threshold. scale=0 makes all
            nonzero pixels foreground; scale=1 uses the raw Otsu threshold.
    """
    if scale < 0:
        raise ValueError("scale must be greater than or equal to 0.")

    gray = to_gray(image)
    threshold = float(threshold_otsu(gray))
    adjusted_threshold = threshold * scale
    return (gray > adjusted_threshold).astype(np.uint8)
