from collections.abc import Callable

import cv2
import numpy as np

from ..core.validate import ensure_image
from .convert import to_uint8


def clahe_norm(
    image: np.ndarray,
    clipLimit: float = 2.0,
    tileGridSize: tuple[int, int] = (8, 8),
) -> np.ndarray:
    image = ensure_image(image, name="image")
    image = to_uint8(image)
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit,
        tileGridSize=tileGridSize,
    )
    return _apply_luminance(image, clahe.apply)


def hist_norm(image: np.ndarray) -> np.ndarray:
    image = ensure_image(image, name="image")
    image = to_uint8(image)
    return _apply_luminance(image, cv2.equalizeHist)


def _apply_luminance(
    image: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if image.ndim == 2:
        return transform(image)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = transform(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def zscore_norm(
    image: np.ndarray,
    std_range: float = 3.0,
) -> np.ndarray:
    image = ensure_image(image, name="image")

    if std_range <= 0:
        raise ValueError("std_range must be greater than 0.")

    norm = image.astype(np.float32)
    work = norm if norm.ndim == 3 else norm[..., None]
    mean = np.mean(work, axis=(0, 1), keepdims=True)
    std = np.std(work, axis=(0, 1), keepdims=True)
    safe_std = np.where(std == 0.0, 1.0, std)

    work -= mean
    work /= safe_std
    np.clip(work, -std_range, std_range, out=work)

    work += std_range
    work *= 255.0 / (2.0 * std_range)
    scaled = np.where(std == 0.0, 127.5, work)

    result = to_uint8(scaled)
    return result if norm.ndim == 3 else result[..., 0]


def minmax_norm(image: np.ndarray) -> np.ndarray:
    image = ensure_image(image, name="image")

    result = cv2.normalize(
        image,
        None,
        alpha=0.0,
        beta=255.0,
        norm_type=cv2.NORM_MINMAX,
    )
    return to_uint8(result)
