import cv2
import numpy as np

from .._core.validate import ensure_image
from ..image.convert import to_gray


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

    image = ensure_image(image, name="image")
    gray = to_gray(image)
    try:
        threshold, _ = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    except Exception as exc:
        raise RuntimeError(f"otsu_threshold failed: {exc}") from exc

    adjusted_threshold = float(threshold) * scale
    return (gray > adjusted_threshold).astype(np.uint8)
