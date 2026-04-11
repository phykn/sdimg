import numpy as np

from .._core.validate import ensure_image
from .convert import to_uint8


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> np.ndarray:
    image = ensure_image(image, name="image")

    brightness_val = np.clip(brightness, -1.0, 1.0)
    contrast_val = np.clip(contrast, -1.0, 1.0)

    adjusted = image.astype(np.float32)

    if brightness_val != 0.0:
        adjusted += brightness_val * 255.0

    factor = 1.0 + contrast_val
    if factor != 1.0:
        adjusted -= 128.0
        adjusted *= factor
        adjusted += 128.0

    return to_uint8(adjusted)
