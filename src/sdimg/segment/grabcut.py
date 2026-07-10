import cv2
import numpy as np

from ..core.types import BBox
from ..core.validation import validate_finite, validate_image, validate_positive_int
from ..image.conversion import convert_to_rgb
from ..mask.conversion import convert_to_mask
from ..mask.distance import compute_distance_transform
from ..mask.geometry import count_foreground
from ..spatial.crop import crop


def refine_grabcut(
    image: np.ndarray,
    roi_mask: np.ndarray,
    bbox: BBox,
    iterations: int = 5,
    margin: int = 20,
    area_tolerance: float = 0.5,
) -> np.ndarray:
    iterations = validate_positive_int(iterations, "iterations")
    if not isinstance(margin, int) or isinstance(margin, bool):
        raise TypeError("margin must be an int.")
    if margin < 0:
        raise ValueError("margin must be greater than or equal to 0.")
    area_tolerance = validate_finite(area_tolerance, "area_tolerance")
    if area_tolerance < 0:
        raise ValueError("area_tolerance must be greater than or equal to 0.")

    image = validate_image(image)
    roi = convert_to_mask(roi_mask)
    cropped = crop(image, bbox)
    if cropped.shape[:2] != roi.shape:
        raise ValueError(
            f"cropped image shape {cropped.shape[:2]} does not match "
            f"roi_mask shape {roi.shape}."
        )
    rgb = convert_to_rgb(cropped)
    original = roi.copy()
    original_area = float(count_foreground(original))
    if original_area == 0 or (margin == 0 and original_area == original.size):
        return original

    try:
        if margin > 0:
            rgb = cv2.copyMakeBorder(
                rgb,
                margin,
                margin,
                margin,
                margin,
                cv2.BORDER_REFLECT,
            )
            roi = cv2.copyMakeBorder(
                roi,
                margin,
                margin,
                margin,
                margin,
                cv2.BORDER_CONSTANT,
                value=0,
            )

        labels = _initialize_labels(roi)
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(
            img=rgb,
            mask=labels,
            rect=None,
            bgdModel=background_model,
            fgdModel=foreground_model,
            iterCount=iterations,
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except Exception as exc:
        raise RuntimeError(f"refine_grabcut failed: {exc}") from exc

    result = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    if margin > 0:
        result = result[margin:-margin, margin:-margin]
    new_area = float(count_foreground(result))
    if abs(new_area - original_area) / original_area > area_tolerance:
        return original
    return np.ascontiguousarray(result)


def _initialize_labels(roi: np.ndarray) -> np.ndarray:
    inside = compute_distance_transform(roi)
    outside = compute_distance_transform((1 - roi).astype(np.uint8))
    threshold = min(float(inside.max()), float(outside.max())) / 5.0

    if threshold <= 0:
        labels = np.full(roi.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        labels[roi == 1] = cv2.GC_PR_FGD
        return labels

    labels = np.full(roi.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[(roi == 0) & (outside < threshold)] = cv2.GC_PR_BGD
    labels[(roi == 1) & (inside < threshold)] = cv2.GC_PR_FGD
    labels[inside >= threshold] = cv2.GC_FGD
    return labels
