import numpy as np

from ..core.validation import validate_image


def prepare_pillow_array(image: np.ndarray) -> tuple[np.ndarray, int]:
    image = validate_image(image)
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if image.ndim == 2:
        return image, 1

    channels = image.shape[2]
    if channels in {1, 2}:
        return image[..., 0], 1
    return image, channels
