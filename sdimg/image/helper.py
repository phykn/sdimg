import numpy as np


def is_image(image: object) -> bool:
    if not isinstance(image, np.ndarray):
        return False

    if image.ndim == 2:
        return True
    if image.ndim != 3:
        return False

    return image.shape[2] in {1, 2, 3, 4}


def to_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError("Image input must be a numpy.ndarray.")

    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=2)
        return to_uint8(rgb)

    if image.ndim != 3:
        raise ValueError("Image input must have shape (H, W) or (H, W, C).")

    channels = image.shape[2]
    if channels < 1 or channels > 4:
        raise ValueError("Image channel count must be one of 1, 2, 3, 4.")

    if channels <= 2:
        rgb = np.repeat(image[..., 0:1], 3, axis=2)
    elif channels == 3:
        rgb = image
    else:
        rgb = image[..., :3]
    return to_uint8(rgb)


def to_gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError("Image input must be a numpy.ndarray.")

    if image.ndim == 2:
        gray = image
    else:
        if image.ndim != 3:
            raise ValueError("Image input must have shape (H, W) or (H, W, C).")

        channels = image.shape[2]
        if channels < 1 or channels > 4:
            raise ValueError("Image channel count must be one of 1, 2, 3, 4.")

        if channels <= 2:
            gray = image[..., 0]
        elif channels == 3:
            gray = image[..., 0] * np.float32(0.299)
            gray += image[..., 1] * np.float32(0.587)
            gray += image[..., 2] * np.float32(0.114)
        else:
            gray = image[..., 0] * np.float32(0.299)
            gray += image[..., 1] * np.float32(0.587)
            gray += image[..., 2] * np.float32(0.114)

    return to_uint8(gray)


def to_uint8(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ValueError("Image input must be a numpy.ndarray.")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        clipped = np.clip(image, 0.0, 255.0)
        return np.rint(clipped).astype(np.uint8)

    return np.clip(image, 0, 255).astype(np.uint8)
