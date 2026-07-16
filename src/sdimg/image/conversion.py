import numpy as np

from ..core.validation import validate_array, validate_image


def is_image(image: object) -> bool:
    try:
        validate_image(image)
    except (TypeError, ValueError):
        return False
    return True


def convert_to_uint8(array: np.ndarray) -> np.ndarray:
    array = validate_array(array)
    if not (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.number)):
        raise ValueError("array must have a numeric or boolean dtype.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("array must have a real-valued dtype.")
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        return np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def convert_to_gray(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    if image.ndim == 2:
        return convert_to_uint8(image)

    if image.shape[2] <= 2:
        return convert_to_uint8(image[..., 0])

    rgb = image[..., :3].astype(np.float32)
    gray = rgb[..., 0] * np.float32(0.299)
    gray += rgb[..., 1] * np.float32(0.587)
    gray += rgb[..., 2] * np.float32(0.114)
    return convert_to_uint8(gray)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    if image.ndim == 2:
        gray = convert_to_uint8(image)
        return np.repeat(gray[..., None], 3, axis=2)

    if image.shape[2] <= 2:
        gray = convert_to_uint8(image[..., 0])
        return np.repeat(gray[..., None], 3, axis=2)
    return convert_to_uint8(image[..., :3])
