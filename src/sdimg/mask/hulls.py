import cv2
import numpy as np
from concave_hull import concave_hull as build_concave_hull

from ..core.validation import validate_finite
from .conversion import convert_to_mask


def fill_convex_hull(mask: np.ndarray) -> np.ndarray:
    binary = convert_to_mask(mask)
    return _fill_component_hulls(binary, concavity=None, length_threshold=0.0)


def fill_concave_hull(
    mask: np.ndarray,
    concavity: float = 2.0,
    length_threshold: float = 0.0,
) -> np.ndarray:
    binary = convert_to_mask(mask)
    concavity = validate_finite(concavity, "concavity")
    length_threshold = validate_finite(length_threshold, "length_threshold")
    if concavity <= 0:
        raise ValueError("concavity must be greater than 0.")
    if length_threshold < 0:
        raise ValueError("length_threshold must be greater than or equal to 0.")
    return _fill_component_hulls(binary, concavity, length_threshold)


def _fill_component_hulls(
    mask: np.ndarray,
    concavity: float | None,
    length_threshold: float,
) -> np.ndarray:
    if not np.any(mask):
        return mask
    try:
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        result = np.zeros_like(mask)
        for label in range(1, count):
            component = (labels == label).astype(np.uint8)
            points = np.column_stack(np.nonzero(component))[:, ::-1].astype(np.int32)
            if points.shape[0] < 3:
                result = np.maximum(result, component)
                continue
            if concavity is None:
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(result, hull, 1)
            else:
                polygon = np.asarray(
                    build_concave_hull(
                        points,
                        concavity=concavity,
                        length_threshold=length_threshold,
                    ),
                    dtype=np.int32,
                )
                if polygon.shape[0] < 3:
                    result = np.maximum(result, component)
                else:
                    cv2.fillPoly(result, [polygon], 1)
            result = np.maximum(result, component)
        return result.astype(np.uint8)
    except Exception as exc:
        name = "fill_convex_hull" if concavity is None else "fill_concave_hull"
        raise RuntimeError(f"{name} failed: {exc}") from exc
