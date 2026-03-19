import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray,
    ksize: tuple[int, int],
    sigmaX: float,
    sigmaY: float = 0.0,
    borderType: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
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
    return cv2.medianBlur(image, ksize)
