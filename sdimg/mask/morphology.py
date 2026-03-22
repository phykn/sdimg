import cv2
import numpy as np

from .pad import pad_1px, unpad_1px


def morphology(
    mask: np.ndarray,
    op: str,
    ksize: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Apply a morphological operation to a binary mask.

    Args:
        op: One of 'open', 'close', 'erode', 'dilate'.
    """
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
    padded = pad_1px(mask)

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

    result = unpad_1px(result)
    return (result > 0).astype(np.uint8)
