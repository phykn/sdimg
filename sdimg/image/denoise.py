import cv2
import numpy as np


def denoise(
    image: np.ndarray,
    h: float = 3.0,
    hColor: float = 3.0,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
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
