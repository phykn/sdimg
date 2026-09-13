import cv2
import numpy as np

from ..core.validation import validate_finite, validate_image, validate_positive_int
from .channels import prepare_visual_alpha, restore_visual_alpha
from .conversion import convert_to_uint8


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> np.ndarray:
    image = validate_image(image)
    brightness = _validate_unit_range(brightness, "brightness")
    contrast = _validate_unit_range(contrast, "contrast")
    visual, alpha, ndim, channels = prepare_visual_alpha(
        image,
        convert_visual=True,
    )

    adjusted = (visual.astype(np.float32) - 127.5) * (1.0 + contrast)
    adjusted += 127.5 + brightness * 255.0
    result = convert_to_uint8(adjusted)
    return restore_visual_alpha(result, alpha, ndim, channels)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    visual, alpha, ndim, channels = prepare_visual_alpha(
        image,
        convert_visual=True,
    )
    try:
        result = _apply_luminance(visual, cv2.equalizeHist)
    except Exception as exc:
        raise RuntimeError(f"equalize_histogram failed: {exc}") from exc
    return restore_visual_alpha(result, alpha, ndim, channels)


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
    visual, alpha, ndim, channels = prepare_visual_alpha(
        image,
        convert_visual=True,
    )
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
        result = _apply_luminance(visual, clahe.apply)
    except Exception as exc:
        raise RuntimeError(f"apply_clahe failed: {exc}") from exc
    return restore_visual_alpha(result, alpha, ndim, channels)


def _apply_luminance(image: np.ndarray, transform: object) -> np.ndarray:
    if image.ndim == 2:
        return transform(image)  # type: ignore[operator]
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = transform(ycrcb[..., 0])  # type: ignore[operator]
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


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
