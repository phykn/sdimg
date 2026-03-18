from typing import Final

import numpy as np


RGB_TO_GRAY_WEIGHTS: Final[np.ndarray] = np.array(
    [0.299, 0.587, 0.114],
    dtype=np.float32,
)


def is_image(src: object) -> bool:
    if not isinstance(src, np.ndarray):
        return False

    try:
        _select_image_channels(src)
    except ValueError:
        return False

    return True


def is_mask(src: object) -> bool:
    if not isinstance(src, np.ndarray):
        return False

    if src.ndim != 2:
        return False

    unique_values = set(np.unique(src).tolist())

    try:
        _normalize_binary_mask(src, unique_values)
    except ValueError:
        return False

    return True


def to_rgb(src: np.ndarray) -> np.ndarray:
    image = _require_array(src)
    channel_view = _select_image_channels(image)
    rgb = _convert_to_rgb(channel_view)
    return _finalize_uint8_image(rgb)


def to_gray(src: np.ndarray) -> np.ndarray:
    image = _require_array(src)
    channel_view = _select_image_channels(image)
    gray = _convert_to_gray(channel_view)
    return _finalize_uint8_image(gray)


def to_mask(src: np.ndarray) -> np.ndarray:
    mask = _require_array(src)

    if mask.ndim != 2:
        raise ValueError("Mask input must have shape (H, W).")

    unique_values = set(np.unique(mask).tolist())
    normalized = _normalize_binary_mask(mask, unique_values)
    return normalized.astype(np.uint8, copy=False)


def _require_array(src: np.ndarray) -> np.ndarray:
    if not isinstance(src, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray.")
    return src


def _select_image_channels(src: np.ndarray) -> np.ndarray:
    if src.ndim == 2:
        return src

    if src.ndim != 3:
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    channel_count = src.shape[2]
    if channel_count not in {1, 2, 3, 4}:
        raise ValueError("Image channel count must be one of 1, 2, 3, 4.")

    if channel_count == 1:
        return src[..., 0]

    if channel_count == 2:
        return src[..., 0]

    if channel_count == 3:
        return src

    return src[..., :3]


def _convert_to_rgb(src: np.ndarray) -> np.ndarray:
    if src.ndim == 2:
        return np.repeat(src[..., None], 3, axis=2)

    return src


def _convert_to_gray(src: np.ndarray) -> np.ndarray:
    if src.ndim == 2:
        return src

    rgb = src.astype(np.float32, copy=False)
    return np.tensordot(rgb, RGB_TO_GRAY_WEIGHTS, axes=([-1], [0]))


def _finalize_uint8_image(src: np.ndarray) -> np.ndarray:
    clipped = np.clip(src.astype(np.float32, copy=False), 0.0, 255.0)
    rounded = np.rint(clipped)
    return rounded.astype(np.uint8)


def _normalize_binary_mask(
    src: np.ndarray,
    unique_values: set[bool] | set[int] | set[float],
) -> np.ndarray:
    if src.dtype == np.bool_:
        return src.astype(np.uint8)

    if unique_values <= {0, 1}:
        return src.astype(np.uint8)

    if unique_values <= {0, 255}:
        return (src > 0).astype(np.uint8)

    if unique_values <= {0.0, 1.0}:
        return src.astype(np.uint8)

    raise ValueError(
        "Mask input must contain only binary values represented as "
        "bool, {0, 1}, {0, 255}, or {0.0, 1.0}.",
    )
