import numpy as np

from .._core.validate import ensure_bbox, ensure_src


def crop(
    src: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    src = ensure_src(src, name="src")
    wmin, hmin, wmax, hmax = ensure_bbox(bbox, shape=src.shape[:2], name="bbox")
    return src[hmin:hmax, wmin:wmax].copy()
