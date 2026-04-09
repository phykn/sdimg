import numpy as np

from .._core.validate import ensure_image, ensure_ndarray


def is_image(image: object) -> bool:
    try:
        ensure_image(image, name="image")
    except (TypeError, ValueError):
        return False
    return True


def to_rgb(image: np.ndarray) -> np.ndarray:
    image = ensure_image(image, name="image")

    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=2)
        return to_uint8(rgb)

    channels = image.shape[2]
    if channels <= 2:
        rgb = np.repeat(image[..., 0:1], 3, axis=2)
    elif channels == 3:
        rgb = image
    else:
        rgb = image[..., :3]
    return to_uint8(rgb)


def to_gray(image: np.ndarray) -> np.ndarray:
    image = ensure_image(image, name="image")

    if image.ndim == 2:
        return to_uint8(image)

    channels = image.shape[2]
    if channels <= 2:
        gray = image[..., 0]
    else:
        gray = image[..., 0] * np.float32(0.299)
        gray += image[..., 1] * np.float32(0.587)
        gray += image[..., 2] * np.float32(0.114)

    return to_uint8(gray)


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = ensure_ndarray(image, name="image")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        clipped = np.clip(image, 0.0, 255.0)
        return np.rint(clipped).astype(np.uint8)

    return np.clip(image, 0, 255).astype(np.uint8)
