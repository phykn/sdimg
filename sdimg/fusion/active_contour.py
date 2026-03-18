import numpy as np
from skimage.draw import polygon2mask
from skimage.segmentation import active_contour

from ..core import is_image, to_gray, to_mask


def active_contour_refine(
    image: np.ndarray,
    initial_mask: np.ndarray,
    num_points: int = 200,
    alpha: float = 0.01,
    beta: float = 0.1,
    gamma: float = 0.01,
    max_num_iter: int = 250,
    convergence: float = 0.1,
) -> np.ndarray:
    if not is_image(image):
        raise ValueError("image must be a valid image ndarray.")

    if num_points < 3:
        raise ValueError("num_points must be greater than or equal to 3.")

    if alpha < 0 or beta < 0 or gamma <= 0:
        raise ValueError("alpha and beta must be >= 0, gamma must be > 0.")

    if max_num_iter <= 0:
        raise ValueError("max_num_iter must be greater than 0.")

    if convergence <= 0:
        raise ValueError("convergence must be greater than 0.")

    gray = to_gray(image).astype(np.float32) / 255.0
    mask = to_mask(initial_mask)

    if gray.shape != mask.shape:
        raise ValueError("image and initial_mask must have the same spatial shape.")

    if np.count_nonzero(mask) == 0:
        return mask

    if np.count_nonzero(mask) == mask.size:
        return mask

    initial_snake = _build_initial_snake(mask, num_points)
    snake = active_contour(
        gray,
        initial_snake,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_num_iter=max_num_iter,
        convergence=convergence,
    )
    refined = polygon2mask(mask.shape, snake)
    return refined.astype(np.uint8)


def _build_initial_snake(
    mask: np.ndarray,
    num_points: int,
) -> np.ndarray:
    coords = np.argwhere(mask == 1)
    hmin = float(coords[:, 0].min())
    hmax = float(coords[:, 0].max())
    wmin = float(coords[:, 1].min())
    wmax = float(coords[:, 1].max())

    center_h = (hmin + hmax) / 2.0
    center_w = (wmin + wmax) / 2.0
    radius_h = max((hmax - hmin + 1.0) / 2.0, 1.0)
    radius_w = max((wmax - wmin + 1.0) / 2.0, 1.0)

    angles = np.linspace(0.0, 2.0 * np.pi, num=num_points, endpoint=False)
    snake_h = center_h + radius_h * np.sin(angles)
    snake_w = center_w + radius_w * np.cos(angles)
    return np.column_stack([snake_h, snake_w]).astype(np.float64)
