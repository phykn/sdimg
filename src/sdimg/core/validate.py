import numpy as np

from .errors import type_error
from .types import BBox


def ensure_ndarray(value: object, name: str = "value") -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise type_error(name=name, expected="a numpy.ndarray")
    return value


def ensure_src(src: object, name: str = "src") -> np.ndarray:
    arr = ensure_ndarray(src, name=name)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"{name} must have shape (H, W) or (H, W, C).")
    return arr


def ensure_image(image: object, name: str = "image") -> np.ndarray:
    """Validate an image array.

    Accepts shape (H, W) or (H, W, C) with C in {1, 2, 3, 4}.

    Channel-count semantics (not runtime-enforced, documented contract):
        C == 1: grayscale
        C == 2: grayscale + alpha (alpha ignored by to_gray/to_rgb)
        C == 3: RGB — sdimg assumes RGB channel order, not BGR
        C == 4: RGBA (alpha ignored by to_gray/to_rgb)
    """
    arr = ensure_src(image, name=name)
    if arr.ndim == 2:
        return arr
    channels = arr.shape[2]
    if channels not in {1, 2, 3, 4}:
        raise ValueError(f"{name} must have shape (H, W) or (H, W, C) with C in 1..4.")
    return arr


def ensure_mask(mask: object, name: str = "mask") -> np.ndarray:
    arr = ensure_ndarray(mask, name=name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W).")
    return arr


def ensure_bbox(
    bbox: BBox,
    shape: tuple[int, int],
    name: str = "bbox",
) -> BBox:
    wmin, hmin, wmax, hmax = bbox
    h, w = shape
    if wmin < 0 or hmin < 0 or wmax > w or hmax > h or wmin >= wmax or hmin >= hmax:
        raise ValueError(f"{name} is out of bounds or invalid.")
    return bbox
