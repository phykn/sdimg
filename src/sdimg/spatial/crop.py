import numpy as np

from ..core.types import BBox
from ..core.validate import ensure_bbox, ensure_src


def crop(
    src: np.ndarray,
    bbox: BBox,
) -> np.ndarray:
    src = ensure_src(src, name="src")
    wmin, hmin, wmax, hmax = ensure_bbox(bbox, shape=src.shape[:2], name="bbox")
    return src[hmin:hmax, wmin:wmax].copy()
