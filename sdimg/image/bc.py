import numpy as np

from ..core import is_image


def adjust_brightness_contrast(
    src: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    brightness_val = np.clip(brightness, -1.0, 1.0)
    contrast_val = np.clip(contrast, -1.0, 1.0)

    adjusted = src.astype(np.float32, copy=False)
    adjusted = adjusted + brightness_val * 255.0

    factor = 1.0 + contrast_val
    adjusted = (adjusted - 128.0) * factor + 128.0

    clipped = np.clip(adjusted, 0.0, 255.0)
    rounded = np.rint(clipped)
    return rounded.astype(np.uint8)
