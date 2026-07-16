from pathlib import Path

import numpy as np
from PIL import Image

from .pillow import prepare_pillow_array


def read_image(path: str | Path) -> np.ndarray:
    path = _validate_path(path)
    try:
        with Image.open(path) as image:
            array = np.array(image)
            if array.dtype == np.uint8:
                return np.array(image.convert("RGB"), dtype=np.uint8)
            return _to_rgb(_scale_to_uint8(array))
    except Exception as exc:
        raise RuntimeError(f"read_image failed: {exc}") from exc


def write_image(path: str | Path, image: np.ndarray, **kwargs: object) -> None:
    path = _validate_path(path)
    pillow_array, _ = prepare_pillow_array(image)
    try:
        Image.fromarray(pillow_array).convert("RGB").save(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"write_image failed: {exc}") from exc


def _validate_path(path: object) -> str | Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or pathlib.Path.")
    return path


def _scale_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.unsignedinteger):
        maximum = np.iinfo(array.dtype).max
        scaled = array.astype(np.float64) * (255.0 / maximum)
        return np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)

    values = array.astype(np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)
    finite_values = values[finite]
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    scaled = np.zeros(array.shape, dtype=np.float64)

    if np.issubdtype(array.dtype, np.floating) and 0.0 <= minimum and maximum <= 1.0:
        scaled[finite] = values[finite] * 255.0
    elif maximum > minimum:
        scaled[finite] = (values[finite] - minimum) * (255.0 / (maximum - minimum))
    else:
        scaled[finite] = np.clip(maximum, 0.0, 255.0)
    return np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.uint8)


def _to_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError("image must have shape (H, W) or (H, W, C).")
    channels = array.shape[2]
    if channels <= 2:
        return np.repeat(array[..., :1], 3, axis=2)
    if channels == 3:
        return array
    if channels == 4:
        return array[..., :3]
    raise ValueError("image must have C in 1..4.")
