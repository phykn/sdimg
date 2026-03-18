import cv2
import numpy as np

from ..core import is_image, to_mask, to_rgb


def grabcut_refine(
    image: np.ndarray,
    initial_mask: np.ndarray,
    iter_count: int = 5,
) -> np.ndarray:
    if not is_image(image):
        raise ValueError("image must be a valid image ndarray.")

    if iter_count <= 0:
        raise ValueError("iter_count must be greater than 0.")

    normalized_image = to_rgb(image)
    normalized_mask = to_mask(initial_mask)

    if normalized_image.shape[:2] != normalized_mask.shape:
        raise ValueError("image and initial_mask must have the same spatial shape.")

    if np.count_nonzero(normalized_mask) == 0:
        return normalized_mask

    if np.count_nonzero(normalized_mask) == normalized_mask.size:
        return normalized_mask

    grabcut_mask = _build_grabcut_mask(normalized_mask)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        normalized_image,
        grabcut_mask,
        None,
        bgd_model,
        fgd_model,
        iter_count,
        cv2.GC_INIT_WITH_MASK,
    )

    refined = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    return refined.astype(np.uint8)


def _build_grabcut_mask(
    mask: np.ndarray,
) -> np.ndarray:
    grabcut_mask = np.full(mask.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[mask == 1] = cv2.GC_PR_FGD
    return grabcut_mask
