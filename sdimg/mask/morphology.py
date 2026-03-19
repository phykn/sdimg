import cv2
import numpy as np

from .pad import pad_1px, unpad_1px


def morphology(
    mask: np.ndarray,
    op: str,
    ksize: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    if op == "open":
        cv2_op = cv2.MORPH_OPEN
    elif op == "close":
        cv2_op = cv2.MORPH_CLOSE
    else:
        raise ValueError("op must be one of: 'open', 'close'.")

    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    padded = pad_1px(mask)
    result = cv2.morphologyEx(
        padded,
        cv2_op,
        kernel,
        iterations=iterations,
    )
    result = unpad_1px(result)
    return result
