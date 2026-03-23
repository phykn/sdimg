import cv2
import numpy as np

from .helper import is_image


def sharpen(
    image: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    if not is_image(image):
        raise ValueError("image must have shape (H, W) or (H, W, C) with C in 1..4.")

    blurred = cv2.GaussianBlur(
        src=image,
        ksize=(0, 0),
        sigmaX=1.0,
        sigmaY=0.0,
        borderType=cv2.BORDER_DEFAULT,
    )

    return cv2.addWeighted(
        src1=image,
        alpha=1.0 + alpha,
        src2=blurred,
        beta=-alpha,
        gamma=0.0,
    )
