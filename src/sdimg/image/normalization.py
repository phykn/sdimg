import numpy as np

from ..core.validation import validate_finite, validate_image
from .conversion import _prepare_visual_alpha, _restore_visual_alpha, convert_to_uint8


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    visual, alpha, ndim, channels = _prepare_visual_alpha(
        image,
        convert_visual=False,
    )
    result = convert_to_uint8(_scale_channels(visual) * 255.0)
    return _restore_visual_alpha(result, alpha, ndim, channels)


def normalize_zscore(image: np.ndarray, std_range: float = 3.0) -> np.ndarray:
    image = validate_image(image)
    std_range = validate_finite(std_range, "std_range")
    if std_range <= 0:
        raise ValueError("std_range must be greater than 0.")
    visual, alpha, ndim, channels = _prepare_visual_alpha(
        image,
        convert_visual=False,
    )
    work = _scale_channels(visual)
    mean = work.mean(axis=(0, 1), keepdims=True)
    std = work.std(axis=(0, 1), keepdims=True)
    scores = np.divide(work - mean, std, out=np.zeros_like(work), where=std != 0)
    scores = np.clip(scores, -std_range, std_range) / std_range
    result = convert_to_uint8((scores + 1.0) * 127.5)
    return _restore_visual_alpha(result, alpha, ndim, channels)


def _scale_channels(visual: np.ndarray) -> np.ndarray:
    work = visual.astype(np.float64)
    if not np.all(np.isfinite(work)):
        raise ValueError("image must contain only finite values for normalization.")

    # Power-of-two scaling bounds arithmetic without rounding nearby large values.
    scale = np.max(np.abs(work), axis=(0, 1), keepdims=True)
    _, exponent = np.frexp(scale)
    work = np.ldexp(work, -exponent)
    work -= work.min(axis=(0, 1), keepdims=True)
    span = work.max(axis=(0, 1), keepdims=True)
    return np.divide(work, span, out=np.full_like(work, 0.5), where=span != 0)
