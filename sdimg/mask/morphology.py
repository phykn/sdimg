import cv2
import numpy as np

from ..core import to_mask


MORPH_OPS = {
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
}


def morphology(
    src: np.ndarray,
    op: str,
    ksize: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    mask = to_mask(src)

    if op not in MORPH_OPS:
        raise ValueError("op must be one of: 'open', 'close'.")

    kernel = np.ones(ksize, dtype=np.uint8)
    result = cv2.morphologyEx(
        mask,
        MORPH_OPS[op],
        kernel,
        iterations=iterations,
    )
    return (result > 0).astype(np.uint8)
