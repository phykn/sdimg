import cv2
import numpy as np

from ..core import is_image, to_gray, to_mask


def guided_filter_refine(
    image: np.ndarray,
    initial_mask: np.ndarray,
    radius: int = 4,
    eps: float = 1e-3,
    threshold: float = 0.5,
) -> np.ndarray:
    if not is_image(image):
        raise ValueError("image must be a valid image ndarray.")

    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0.")

    if eps <= 0:
        raise ValueError("eps must be greater than 0.")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must satisfy 0 <= threshold <= 1.")

    guide = to_gray(image).astype(np.float32) / 255.0
    mask = to_mask(initial_mask).astype(np.float32)

    if guide.shape != mask.shape:
        raise ValueError("image and initial_mask must have the same spatial shape.")

    refined = _guided_filter(guide, mask, radius, eps)
    return (refined >= threshold).astype(np.uint8)


def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    kernel = (2 * radius + 1, 2 * radius + 1)

    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, kernel, normalize=True)
    mean_src = cv2.boxFilter(src, cv2.CV_32F, kernel, normalize=True)
    corr_guide = cv2.boxFilter(guide * guide, cv2.CV_32F, kernel, normalize=True)
    corr_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, kernel, normalize=True)

    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel, normalize=True)
    refined = mean_a * guide + mean_b
    return np.clip(refined, 0.0, 1.0)
