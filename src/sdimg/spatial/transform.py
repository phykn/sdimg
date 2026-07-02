import numpy as np
from typing import Literal

from ..core.validate import ensure_src

Rotation = Literal[0, 90, 180, 270]
FlipDirection = Literal["horizontal", "vertical", "transpose"]


def rotate(
    src: np.ndarray,
    rotation: Rotation = 0,
) -> np.ndarray:
    src = ensure_src(src, name="src")

    if rotation not in {0, 90, 180, 270}:
        raise ValueError("rotation must be one of 0, 90, 180, 270.")
    return np.rot90(src, k=rotation // 90)


def flip(
    src: np.ndarray,
    direction: FlipDirection,
) -> np.ndarray:
    src = ensure_src(src, name="src")

    if direction == "horizontal":
        return np.flip(src, axis=1)
    if direction == "vertical":
        return np.flip(src, axis=0)
    if direction == "transpose":
        return np.swapaxes(src, 0, 1)
    raise ValueError("direction must be one of horizontal, vertical, transpose.")
