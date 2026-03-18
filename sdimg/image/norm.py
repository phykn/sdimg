import cv2
import numpy as np

from ..core import is_image


def clahe_norm(
    src: np.ndarray,
    clipLimit: float = 40.0,
    tileGridSize: tuple[int, int] = (8, 8),
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    clahe = cv2.createCLAHE(
        clipLimit=clipLimit,
        tileGridSize=tileGridSize,
    )
    image = src.astype(np.uint8, copy=False)

    if image.ndim == 2:
        return clahe.apply(image)

    channels = []
    for idx in range(image.shape[2]):
        channel = image[..., idx]
        channels.append(clahe.apply(channel))

    return np.stack(channels, axis=2)


def hist_norm(src: np.ndarray) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    image = src.astype(np.uint8, copy=False)

    if image.ndim == 2:
        return cv2.equalizeHist(image)

    channels = []
    for idx in range(image.shape[2]):
        channel = image[..., idx]
        channels.append(cv2.equalizeHist(channel))

    return np.stack(channels, axis=2)


def standard_norm(
    src: np.ndarray,
    std_range: float = 3.0,
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    if std_range <= 0:
        raise ValueError("std_range must be greater than 0.")

    image = src.astype(np.float32, copy=False)

    if image.ndim == 2:
        normalized = _normalize_standard_channel(image, std_range)
    else:
        channels = []
        for idx in range(image.shape[2]):
            channel = image[..., idx]
            channels.append(_normalize_standard_channel(channel, std_range))
        normalized = np.stack(channels, axis=2)

    clipped = np.clip(normalized, 0.0, 255.0)
    rounded = np.rint(clipped)
    return rounded.astype(np.uint8)


def _normalize_standard_channel(
    src: np.ndarray,
    std_range: float,
) -> np.ndarray:
    mean_val = float(np.mean(src))
    std_val = float(np.std(src))

    if std_val == 0.0:
        return src

    lower = mean_val - std_range * std_val
    upper = mean_val + std_range * std_val
    clipped = np.clip(src, lower, upper)

    return (clipped - lower) / (upper - lower) * 255.0
