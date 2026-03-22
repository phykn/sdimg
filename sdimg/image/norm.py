import cv2
import numpy as np

from .helper import to_uint8


def clahe_norm(
    image: np.ndarray,
    clipLimit: float = 40.0,
    tileGridSize: tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clipLimit,
        tileGridSize=tileGridSize,
    )
    if image.ndim == 2:
        return clahe.apply(image)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def hist_norm(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def zscore_norm(
    image: np.ndarray,
    std_range: float = 3.0,
) -> np.ndarray:
    if std_range <= 0:
        raise ValueError("std_range must be greater than 0.")

    norm = image.astype(np.float32)
    work = norm if norm.ndim == 3 else norm[..., None]
    mean = np.mean(work, axis=(0, 1), keepdims=True)
    std = np.std(work, axis=(0, 1), keepdims=True)
    safe_std = np.where(std == 0.0, 1.0, std)

    zscore = (work - mean) / safe_std
    clipped = np.clip(zscore, -std_range, std_range)
    scaled = (clipped + std_range) / (2.0 * std_range) * 255.0
    scaled = np.where(std == 0.0, 127.5, scaled)

    result = to_uint8(scaled)
    return result if norm.ndim == 3 else result[..., 0]


def minmax_norm(image: np.ndarray) -> np.ndarray:
    result = cv2.normalize(
        image,
        None,
        alpha=0.0,
        beta=255.0,
        norm_type=cv2.NORM_MINMAX,
    )
    return to_uint8(result)
