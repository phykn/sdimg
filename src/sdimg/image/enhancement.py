import cv2
import numpy as np

from ..core.validation import validate_finite, validate_image, validate_positive_int
from .conversion import (
    _restore_visual_alpha,
    _split_visual_alpha,
    convert_to_uint8,
)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> np.ndarray:
    image = validate_image(image)
    brightness = _validate_unit_range(brightness, "brightness")
    contrast = _validate_unit_range(contrast, "contrast")
    visual, alpha, ndim, channels = _split_image(image, convert_visual=True)

    adjusted = (visual.astype(np.float32) - 127.5) * (1.0 + contrast)
    adjusted += 127.5 + brightness * 255.0
    result = convert_to_uint8(adjusted)
    return _restore_visual_alpha(result, alpha, ndim, channels)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    visual, alpha, ndim, channels = _split_image(image, convert_visual=True)
    try:
        result = _apply_luminance(visual, cv2.equalizeHist)
    except Exception as exc:
        raise RuntimeError(f"equalize_histogram failed: {exc}") from exc
    return _restore_visual_alpha(result, alpha, ndim, channels)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    image = validate_image(image)
    clip_limit = validate_finite(clip_limit, "clip_limit")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than 0.")
    grid = _validate_grid_size(tile_grid_size)
    visual, alpha, ndim, channels = _split_image(image, convert_visual=True)
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
        result = _apply_luminance(visual, clahe.apply)
    except Exception as exc:
        raise RuntimeError(f"apply_clahe failed: {exc}") from exc
    return _restore_visual_alpha(result, alpha, ndim, channels)


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    visual, alpha, ndim, channels = _split_image(image, convert_visual=False)
    result = _normalize_channels(visual, method="minmax", std_range=0.0)
    return _restore_visual_alpha(result, alpha, ndim, channels)


def normalize_zscore(image: np.ndarray, std_range: float = 3.0) -> np.ndarray:
    image = validate_image(image)
    std_range = validate_finite(std_range, "std_range")
    if std_range <= 0:
        raise ValueError("std_range must be greater than 0.")
    visual, alpha, ndim, channels = _split_image(image, convert_visual=False)
    result = _normalize_channels(visual, method="zscore", std_range=std_range)
    return _restore_visual_alpha(result, alpha, ndim, channels)


def _split_image(
    image: np.ndarray,
    *,
    convert_visual: bool,
) -> tuple[np.ndarray, np.ndarray | None, int, int | None]:
    visual, alpha = _split_visual_alpha(image)
    if convert_visual:
        visual = convert_to_uint8(visual)
    if alpha is not None:
        alpha = convert_to_uint8(alpha)
    channels = image.shape[2] if image.ndim == 3 else None
    return visual, alpha, image.ndim, channels


def _apply_luminance(image: np.ndarray, transform: object) -> np.ndarray:
    if image.ndim == 2:
        return transform(image)  # type: ignore[operator]
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = transform(ycrcb[..., 0])  # type: ignore[operator]
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _normalize_channels(
    visual: np.ndarray,
    *,
    method: str,
    std_range: float,
) -> np.ndarray:
    work = visual.astype(np.float64)
    if not np.all(np.isfinite(work)):
        raise ValueError("image must contain only finite values for normalization.")
    channels = work[..., None] if work.ndim == 2 else work
    out = np.empty(channels.shape, dtype=np.uint8)

    for index in range(channels.shape[2]):
        channel = channels[..., index]
        if method == "minmax":
            minimum = float(channel.min())
            maximum = float(channel.max())
            if maximum == minimum:
                out[..., index] = 128
            else:
                out[..., index] = convert_to_uint8(
                    (channel - minimum) * (255.0 / (maximum - minimum)),
                )
        else:
            mean = float(channel.mean())
            std = float(channel.std())
            if std == 0.0:
                out[..., index] = 128
            else:
                normalized = np.clip((channel - mean) / std, -std_range, std_range)
                normalized = (normalized + std_range) * (255.0 / (2.0 * std_range))
                out[..., index] = convert_to_uint8(normalized)

    return out[..., 0] if work.ndim == 2 else out


def _validate_unit_range(value: object, name: str) -> float:
    result = validate_finite(value, name)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between -1 and 1.")
    return result


def _validate_grid_size(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("tile_grid_size must be a tuple of two ints.")
    return (
        validate_positive_int(value[0], "tile_grid_size[0]"),
        validate_positive_int(value[1], "tile_grid_size[1]"),
    )
