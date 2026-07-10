import cv2
import numpy as np

from ..core.validation import validate_finite, validate_image
from ..image.conversion import convert_to_gray


def threshold_otsu(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    image = validate_image(image)
    scale = validate_finite(scale, "scale")
    if scale < 0:
        raise ValueError("scale must be greater than or equal to 0.")
    gray = convert_to_gray(image)
    try:
        threshold, _ = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    except Exception as exc:
        raise RuntimeError(f"threshold_otsu failed: {exc}") from exc
    return (gray > float(threshold) * scale).astype(np.uint8)
