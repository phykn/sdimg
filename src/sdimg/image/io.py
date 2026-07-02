from pathlib import Path

import numpy as np
from PIL import Image

from .._core.validate import ensure_ndarray


def imread(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            array = np.array(image)
            if array.dtype == np.uint8:
                return np.array(image.convert("RGB"))
            return _to_rgb_uint8(_scale_to_uint8(array))
    except Exception as exc:
        raise RuntimeError(f"imread failed: {exc}") from exc


def imwrite(path: str | Path, image: np.ndarray, **kwargs: object) -> None:
    image = ensure_ndarray(image, name="image")
    image_for_pillow = _prepare_write_image(image)

    try:
        Image.fromarray(image_for_pillow).convert("RGB").save(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"imwrite failed: {exc}") from exc


def _prepare_write_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(
            "image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
        )

    channels = image.shape[2]
    if channels == 1:
        return image[..., 0]
    if channels in {3, 4}:
        return image
    raise ValueError(
        "image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
    )


def _scale_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.unsignedinteger):
        max_value = np.iinfo(array.dtype).max
        scaled = array.astype(np.float64) * (255.0 / max_value)
        return np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)

    values = array.astype(np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)

    finite_values = values[finite]
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    scaled = np.zeros(array.shape, dtype=np.float64)

    if 0.0 <= min_value and max_value <= 1.0:
        scaled[finite] = values[finite] * 255.0
    elif max_value > min_value:
        scaled[finite] = (values[finite] - min_value) * (
            255.0 / (max_value - min_value)
        )
    else:
        scaled[finite] = np.clip(max_value, 0.0, 255.0)

    return np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)


def _to_rgb_uint8(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError("image must have shape (H, W) or (H, W, C).")

    channels = array.shape[2]
    if channels <= 2:
        return np.repeat(array[..., 0:1], 3, axis=2)
    if channels == 3:
        return array
    return array[..., :3]
