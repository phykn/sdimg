import cv2
import numpy as np
from concave_hull import concave_hull as build_concave_hull

from ..core import to_mask
from .component import keep_largest_component
from .edge import extract_edge


def concave_hull(
    src: np.ndarray,
    concavity: float = 2.0,
    length_threshold: float = 0.0,
) -> np.ndarray:
    mask = keep_largest_component(src)

    if np.count_nonzero(mask) == 0:
        return mask

    edge = extract_edge(mask)
    points = np.column_stack(np.nonzero(edge))[:, ::-1]

    if points.shape[0] < 3:
        return mask.astype(np.uint8, copy=False)

    hull = build_concave_hull(
        points,
        concavity=concavity,
        length_threshold=length_threshold,
    )
    hull = np.asarray(hull, dtype=np.int32)

    result = np.zeros_like(mask)
    cv2.fillPoly(result, [hull], 1)
    return result.astype(np.uint8, copy=False)


def convex_hull(src: np.ndarray) -> np.ndarray:
    mask = to_mask(src)

    if np.count_nonzero(mask) == 0:
        return mask

    points = np.column_stack(np.nonzero(mask))[:, ::-1]
    hull = cv2.convexHull(points.astype(np.int32))

    result = np.zeros_like(mask)
    cv2.fillConvexPoly(result, hull, 1)
    return result.astype(np.uint8, copy=False)
