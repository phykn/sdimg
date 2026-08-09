from typing import Literal

import numpy as np

from ..core.validation import validate_source

Rotation = Literal[0, 90, 180, 270]
FlipDirection = Literal["horizontal", "vertical", "transpose"]


def rotate(array: np.ndarray, degrees: Rotation = 0) -> np.ndarray:
    array = validate_source(array)
    if not isinstance(degrees, int) or isinstance(degrees, bool):
        raise TypeError("degrees must be an int.")
    if degrees not in {0, 90, 180, 270}:
        raise ValueError("degrees must be one of 0, 90, 180, 270.")
    return np.rot90(array, k=degrees // 90).copy(order="C")


def flip(array: np.ndarray, direction: FlipDirection) -> np.ndarray:
    array = validate_source(array)
    if not isinstance(direction, str):
        raise TypeError("direction must be a str.")
    if direction == "horizontal":
        result = np.flip(array, axis=1)
    elif direction == "vertical":
        result = np.flip(array, axis=0)
    elif direction == "transpose":
        result = np.swapaxes(array, 0, 1)
    else:
        raise ValueError("direction must be one of horizontal, vertical, transpose.")
    return result.copy(order="C")
