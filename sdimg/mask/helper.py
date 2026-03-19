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
    if mask.ndim != 2:
        raise ValueError("Mask input must have shape (H, W).")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    unique_values = set(np.unique(mask).tolist())
    if unique_values <= {0, 1}:
        return mask.astype(np.uint8)

    if unique_values <= {0, 255}:
        return (mask > 0).astype(np.uint8)

    if unique_values <= {0.0, 1.0}:
        return mask.astype(np.uint8)

    raise ValueError(
        "Mask input must contain only binary values represented as "
        "bool, {0, 1}, {0, 255}, or {0.0, 1.0}.",
    )


def get_coords(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask == 1)
    return coords


def get_bbox(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    coords = get_coords(mask)

    if coords.shape[0] == 0:
        return None

    hmin = int(coords[:, 0].min())
    hmax = int(coords[:, 0].max()) + 1
    wmin = int(coords[:, 1].min())
    wmax = int(coords[:, 1].max()) + 1
    return (wmin, hmin, wmax, hmax)


def get_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def get_centroid(
    mask: np.ndarray,
) -> tuple[float, float] | None:
    coords = get_coords(mask)

    if coords.shape[0] == 0:
        return None

    center_h = float(np.mean(coords[:, 0]))
    center_w = float(np.mean(coords[:, 1]))
    return (center_h, center_w)
