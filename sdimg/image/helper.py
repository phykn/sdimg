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
            rgb = image.astype(np.float32, copy=False)
            gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        else:
            rgb = image[..., :3].astype(np.float32, copy=False)
            gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114

    return to_uint8(gray)


def to_uint8(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image.astype(np.float32, copy=False), 0.0, 255.0)
    rounded = np.rint(clipped)
    return rounded.astype(np.uint8)