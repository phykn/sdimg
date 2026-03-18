import numpy as np

from ..core import is_image, is_mask, to_mask


def _require_spatial_input(src: np.ndarray) -> np.ndarray:
    if is_mask(src):
        return to_mask(src)

    if not is_image(src):
        raise ValueError("Input must be a valid image or mask ndarray.")

    return src.astype(np.uint8, copy=False)


def _finalize_image(src: np.ndarray) -> np.ndarray:
    clipped = np.clip(src.astype(np.float32, copy=False), 0.0, 255.0)
    rounded = np.rint(clipped)
    return rounded.astype(np.uint8)


def _finalize_mask(src: np.ndarray) -> np.ndarray:
    mask = src.astype(np.float32, copy=False)
    return (mask >= 0.5).astype(np.uint8)
