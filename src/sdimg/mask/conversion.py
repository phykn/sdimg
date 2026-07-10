import numpy as np

from ..core.validation import validate_mask


def is_mask(mask: object) -> bool:
    try:
        convert_to_mask(mask)
    except (TypeError, ValueError):
        return False
    return True


def convert_to_mask(mask: object) -> np.ndarray:
    array = validate_mask(mask)
    if not (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.number)):
        raise ValueError("mask must have a numeric or boolean dtype.")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("mask must have a real-valued dtype.")
    if array.dtype == np.bool_:
        return array.astype(np.uint8)

    valid_01 = bool(np.all((array == 0) | (array == 1)))
    if valid_01:
        return array.astype(np.uint8)
    valid_0255 = bool(np.all((array == 0) | (array == 255)))
    if valid_0255:
        return (array > 0).astype(np.uint8)
    raise ValueError(
        "mask must contain only binary values represented as bool, {0, 1}, or {0, 255}."
    )
