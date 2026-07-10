import numpy as np

from .types import BBox


def validate_array(value: object, name: str = "array") -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    return value


def validate_source(value: object, name: str = "array") -> np.ndarray:
    array = validate_array(value, name=name)
    if (
        array.ndim not in {2, 3}
        or 0 in array.shape[:2]
        or (array.ndim == 3 and array.shape[2] == 0)
    ):
        raise ValueError(
            f"{name} must have shape (H, W) or (H, W, C) with non-empty dimensions."
        )
    if not (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.number)):
        raise ValueError(f"{name} must have a numeric or boolean dtype.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must have a real-valued dtype.")
    return array


def validate_image(value: object, name: str = "image") -> np.ndarray:
    image = validate_source(value, name=name)
    if image.ndim == 3 and image.shape[2] not in {1, 2, 3, 4}:
        raise ValueError(f"{name} must have shape (H, W) or (H, W, C) with C in 1..4.")
    return image


def validate_mask(value: object, name: str = "mask") -> np.ndarray:
    mask = validate_array(value, name=name)
    if mask.ndim != 2 or 0 in mask.shape:
        raise ValueError(f"{name} must have non-empty shape (H, W).")
    return mask


def validate_bbox(
    bbox: object,
    shape: tuple[int, int] | None = None,
    name: str = "bbox",
) -> BBox:
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        raise TypeError(f"{name} must be a tuple of four ints.")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in bbox
    ):
        raise TypeError(f"{name} must be a tuple of four ints.")

    xmin, ymin, xmax, ymax = bbox
    if xmin < 0 or ymin < 0 or xmin >= xmax or ymin >= ymax:
        raise ValueError(f"{name} has invalid coordinate order.")
    if shape is not None and (xmax > shape[1] or ymax > shape[0]):
        raise ValueError(f"{name} is outside array bounds.")
    return (xmin, ymin, xmax, ymax)


def validate_finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value, bool
    ):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def validate_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value
