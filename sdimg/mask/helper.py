import numpy as np


def is_mask(mask: object) -> bool:
    if not isinstance(mask, np.ndarray):
        return False

    try:
        to_mask(mask)
    except ValueError:
        return False

    return True


def to_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a 2D array to a binary uint8 mask with values in {0, 1}.

    Accepted input formats: bool, {0, 1}, or {0, 255}.
    """
    if mask.ndim != 2:
        raise ValueError("Mask input must have shape (H, W).")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    unique_values = set(np.unique(mask).tolist())
    if unique_values <= {0, 1}:
        return mask.astype(np.uint8)

    if unique_values <= {0, 255}:
        return (mask > 0).astype(np.uint8)

    raise ValueError(
        "Mask input must contain only binary values represented as "
        "bool, {0, 1}, or {0, 255}.",
    )


def get_coords(
    mask: np.ndarray,
    transpose: bool = False,
) -> np.ndarray:
    coords = np.argwhere(mask > 0)

    if transpose:
        return coords.T

    return coords


def get_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of nonzero pixels as (wmin, hmin, wmax, hmax), or None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows):
        return None

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    hmin = int(row_indices[0])
    hmax = int(row_indices[-1]) + 1
    wmin = int(col_indices[0])
    wmax = int(col_indices[-1]) + 1
    return (wmin, hmin, wmax, hmax)


def get_roi_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def get_box_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0

    wmin, hmin, wmax, hmax = bbox
    return (wmax - wmin) * (hmax - hmin)


def get_centroid(
    mask: np.ndarray,
) -> tuple[float, float] | None:
    coords = get_coords(mask)

    if coords.shape[0] == 0:
        return None

    center_h = float(np.mean(coords[:, 0]))
    center_w = float(np.mean(coords[:, 1]))
    return (center_h, center_w)
