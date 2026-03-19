import numpy as np


def rotate(
    src: np.ndarray,
    rotation: int = 0,
) -> np.ndarray:
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("rotation must be one of 0, 90, 180, 270.")
    return np.rot90(src, k=rotation // 90)


def flip(
    src: np.ndarray,
    direction: str,
) -> np.ndarray:
    if direction == "horizontal":
        return np.flip(src, axis=1)
    if direction == "vertical":
        return np.flip(src, axis=0)
    if direction == "transpose":
        return np.swapaxes(src, 0, 1)
    raise ValueError("direction must be one of horizontal, vertical, transpose.")
