import cv2
import numpy as np

from ..core import is_image


def gaussian_blur(
    src: np.ndarray,
    ksize: tuple[int, int],
    sigmaX: float,
    sigmaY: float = 0.0,
    borderType: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    image = src.astype(np.uint8, copy=False)
    return cv2.GaussianBlur(
        image,
        ksize,
        sigmaX,
        sigmaY=sigmaY,
        borderType=borderType,
    )


def median_blur(
    src: np.ndarray,
    ksize: int,
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    image = src.astype(np.uint8, copy=False)
    return cv2.medianBlur(image, ksize)
