import cv2
import numpy as np

from ..core import is_image


def sharpen(
    src: np.ndarray,
    alpha: float = 1.0,
    ksize: tuple[int, int] = (5, 5),
    sigmaX: float = 1.0,
    sigmaY: float = 0.0,
    borderType: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    image = src.astype(np.uint8, copy=False)
    blurred = cv2.GaussianBlur(
        image,
        ksize,
        sigmaX,
        sigmaY=sigmaY,
        borderType=borderType,
    )
    sharpened = cv2.addWeighted(
        image,
        1.0 + alpha,
        blurred,
        -alpha,
        0.0,
    )
    return sharpened.astype(np.uint8, copy=False)
