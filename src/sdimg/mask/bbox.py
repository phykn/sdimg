import numpy as np

from .._core.types import BBox
from .._core.validate import ensure_ndarray
from .convert import to_mask


def get_coords(
    mask: np.ndarray,
    transpose: bool = False,
) -> np.ndarray:
    mask = to_mask(mask)
    coords = _coords_from_mask(mask)

    if transpose:
        return coords.T

    return coords


def get_box_from_coords(
    coords: np.ndarray,
) -> BBox | None:
    coords = ensure_ndarray(coords, name="coords")

    if coords.size == 0:
        return None

    if coords.ndim != 2:
        raise ValueError("coords must have shape (N, 2) or (2, N).")

    if coords.shape[1] == 2:
        h_coords = coords[:, 0]
        w_coords = coords[:, 1]
    elif coords.shape[0] == 2:
        h_coords = coords[0, :]
        w_coords = coords[1, :]
    else:
        raise ValueError("coords must have shape (N, 2) or (2, N).")

    hmin = int(np.min(h_coords))
    hmax = int(np.max(h_coords)) + 1
    wmin = int(np.min(w_coords))
    wmax = int(np.max(w_coords)) + 1
    return (wmin, hmin, wmax, hmax)


def get_box_from_mask(
    mask: np.ndarray,
) -> BBox | None:
    mask = to_mask(mask)
    return _box_from_mask(mask)


def to_roi_box(
    mask: np.ndarray,
) -> dict[str, object] | None:
    mask = to_mask(mask)
    bbox = _box_from_mask(mask)
    if bbox is None:
        return None

    wmin, hmin, wmax, hmax = bbox
    roi = mask[hmin:hmax, wmin:wmax].copy()
    return {"roi": roi, "box": bbox}


def get_roi_size(mask: np.ndarray) -> int:
    mask = to_mask(mask)
    return int(np.count_nonzero(mask))


def get_box_size(bbox: BBox | None) -> int:
    if bbox is None:
        return 0

    wmin, hmin, wmax, hmax = bbox
    if wmin >= wmax or hmin >= hmax:
        raise ValueError("bbox must satisfy wmin < wmax and hmin < hmax.")
    return (wmax - wmin) * (hmax - hmin)


def get_centroid(
    mask: np.ndarray,
) -> tuple[float, float] | None:
    mask = to_mask(mask)
    coords = _coords_from_mask(mask)

    if coords.shape[0] == 0:
        return None

    center_h = float(np.mean(coords[:, 0]))
    center_w = float(np.mean(coords[:, 1]))
    return (center_h, center_w)


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask > 0)


def _box_from_mask(mask: np.ndarray) -> BBox | None:
    return get_box_from_coords(_coords_from_mask(mask))
