import cv2
import numpy as np

from ..core.types import BBox
from ..core.validation import validate_array, validate_bbox
from .conversion import convert_to_mask


def find_foreground_points(mask: np.ndarray) -> np.ndarray:
    binary = convert_to_mask(mask)
    return np.ascontiguousarray(np.argwhere(binary > 0)[:, ::-1])


def find_bbox_from_points(points: np.ndarray) -> BBox | None:
    points = validate_array(points, name="points")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    if points.dtype == np.bool_ or not np.issubdtype(points.dtype, np.integer):
        raise ValueError("points must contain non-negative integer coordinates.")
    if points.shape[0] == 0:
        return None
    if np.any(points < 0):
        raise ValueError("points must contain non-negative integer coordinates.")

    xmin = int(points[:, 0].min())
    ymin = int(points[:, 1].min())
    xmax = int(points[:, 0].max()) + 1
    ymax = int(points[:, 1].max()) + 1
    return (xmin, ymin, xmax, ymax)


def find_bbox(mask: np.ndarray) -> BBox | None:
    return _find_bbox(convert_to_mask(mask), "find_bbox")


def find_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    binary = convert_to_mask(mask)
    try:
        moments = cv2.moments(binary, binaryImage=True)
    except Exception as exc:
        raise RuntimeError(f"find_centroid failed: {exc}") from exc
    area = moments["m00"]
    if area == 0.0:
        return None
    return (float(moments["m10"] / area), float(moments["m01"] / area))


def count_foreground(mask: np.ndarray) -> int:
    return int(np.count_nonzero(convert_to_mask(mask)))


def measure_bbox_area(bbox: BBox | None) -> int:
    if bbox is None:
        return 0
    xmin, ymin, xmax, ymax = validate_bbox(bbox)
    return (xmax - xmin) * (ymax - ymin)


def extract_roi(mask: np.ndarray) -> tuple[np.ndarray, BBox] | None:
    binary = convert_to_mask(mask)
    bbox = _find_bbox(binary, "extract_roi")
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    return binary[ymin:ymax, xmin:xmax].copy(), bbox


def _find_bbox(binary: np.ndarray, function_name: str) -> BBox | None:
    try:
        xmin, ymin, width, height = cv2.boundingRect(binary)
    except Exception as exc:
        raise RuntimeError(f"{function_name} failed: {exc}") from exc
    if width == 0 or height == 0:
        return None
    return (xmin, ymin, xmin + width, ymin + height)
