import numpy as np

from .helper import to_uint8


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> np.ndarray:
    brightness_val = np.clip(brightness, -1.0, 1.0)
    contrast_val = np.clip(contrast, -1.0, 1.0)

    adjusted = image.astype(np.float32, copy=False)
    adjusted = adjusted + brightness_val * 255.0

    factor = 1.0 + contrast_val
    adjusted = (adjusted - 128.0) * factor + 128.0

    return to_uint8(adjusted)
