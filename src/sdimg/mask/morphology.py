import cv2
import numpy as np
from typing import Literal

from .convert import to_mask

MorphologyOp = Literal["open", "close", "erode", "dilate"]


def morphology(
    mask: np.ndarray,
    op: MorphologyOp,
    ksize: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    mask = to_mask(mask)

    ops = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": "erode",
        "dilate": "dilate",
    }
    if op not in ops:
        raise ValueError(f"op must be one of: {', '.join(repr(k) for k in ops)}.")

    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)

    if op == "erode":
        result = cv2.erode(padded, kernel, iterations=iterations)
    elif op == "dilate":
        result = cv2.dilate(padded, kernel, iterations=iterations)
    else:
        result = cv2.morphologyEx(
            padded,
            ops[op],
            kernel,
            iterations=iterations,
        )

    result = result[1:-1, 1:-1]
    return (result > 0).astype(np.uint8)
