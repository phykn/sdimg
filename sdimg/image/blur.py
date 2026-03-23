import cv2
import numpy as np

from .helper import is_image


def gaussian_blur(
    image: np.ndarray,
    ksize: tuple[int, int],
    sigmaX: float,
    sigmaY: float = 0.0,
    borderType: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    if not is_image(image):
        raise ValueError("image must have shape (H, W) or (H, W, C) with C in 1..4.")

    return cv2.GaussianBlur(
        image,
        ksize,
        sigmaX,
        sigmaY=sigmaY,
        borderType=borderType,
    )


def median_blur(
    image: np.ndarray,
    ksize: int,
) -> np.ndarray:
    if not is_image(image):
        raise ValueError("image must have shape (H, W) or (H, W, C) with C in 1..4.")

    return cv2.medianBlur(image, ksize)
