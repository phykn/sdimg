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
    visual, _ = _split_visual_alpha(image)
    if visual.ndim == 2:
        return convert_to_uint8(visual)

    wide = np.issubdtype(visual.dtype, np.floating) and np.any(
        np.abs(visual) > np.finfo(np.float32).max
    )
    dtype = np.float64 if wide else np.float32
    rgb = visual.astype(dtype)
    gray = rgb[..., 0] * dtype(0.299)
    gray += rgb[..., 1] * dtype(0.587)
    gray += rgb[..., 2] * dtype(0.114)
    return convert_to_uint8(gray)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    image = validate_image(image)
    visual, _ = _split_visual_alpha(image)
    if visual.ndim == 2:
        gray = convert_to_uint8(visual)
        return np.repeat(gray[..., None], 3, axis=2)
    return convert_to_uint8(visual)


def _split_visual_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 2:
        return image, None
    channels = image.shape[2]
    visual = image[..., 0] if channels <= 2 else image[..., :3]
    alpha = image[..., -1] if channels in {2, 4} else None
    return visual, alpha


def _prepare_visual_alpha(
    image: np.ndarray,
    convert_visual: bool,
) -> tuple[np.ndarray, np.ndarray | None, int, int | None]:
    visual, alpha = _split_visual_alpha(image)
    if convert_visual:
        visual = convert_to_uint8(visual)
    if alpha is not None:
        alpha = convert_to_uint8(alpha)
    channels = image.shape[2] if image.ndim == 3 else None
    return visual, alpha, image.ndim, channels


def _restore_visual_alpha(
    visual: np.ndarray,
    alpha: np.ndarray | None,
    ndim: int,
    channels: int | None,
) -> np.ndarray:
    if ndim == 2:
        return np.ascontiguousarray(visual)
    if channels == 1:
        return np.ascontiguousarray(visual[..., None])
    if alpha is None:
        return np.ascontiguousarray(visual)
    if visual.ndim == 2:
        visual = visual[..., None]
    return np.ascontiguousarray(np.concatenate([visual, alpha[..., None]], axis=2))


def _prepare_pillow_array(image: np.ndarray) -> tuple[np.ndarray, int]:
    image = validate_image(image)
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if image.ndim == 3 and image.shape[2] == 4:
        return image, 4
    visual, _ = _split_visual_alpha(image)
    return visual, 1 if visual.ndim == 2 else 3
