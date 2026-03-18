import cv2
import numpy as np

from ..core import is_image


def denoise(
    src: np.ndarray,
    h: float = 3.0,
    hColor: float = 3.0,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
) -> np.ndarray:
    if not is_image(src):
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    image = src.astype(np.uint8, copy=False)

    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(
            image,
            h=h,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize,
        )

    if image.shape[2] == 3:
        return cv2.fastNlMeansDenoisingColored(
            image,
            h=h,
            hColor=hColor,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize,
        )

    return cv2.fastNlMeansDenoising(
        image,
        h=h,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize,
    )
