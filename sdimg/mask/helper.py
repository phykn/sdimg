import numpy as np

from ..core import to_mask


def get_coords(src: np.ndarray) -> np.ndarray:
    mask = to_mask(src)
    coords = np.argwhere(mask == 1)
    return coords.astype(np.int64, copy=False)


def get_bbox(
    src: np.ndarray,
) -> tuple[int, int, int, int] | None:
    coords = get_coords(src)

    if coords.shape[0] == 0:
        return None

    hmin = int(coords[:, 0].min())
    hmax = int(coords[:, 0].max()) + 1
    wmin = int(coords[:, 1].min())
    wmax = int(coords[:, 1].max()) + 1
    return (wmin, hmin, wmax, hmax)


def get_area(src: np.ndarray) -> int:
    mask = to_mask(src)
    return int(np.count_nonzero(mask))


def get_centroid(
    src: np.ndarray,
) -> tuple[float, float] | None:
    coords = get_coords(src)

    if coords.shape[0] == 0:
        return None

    center_h = float(np.mean(coords[:, 0]))
    center_w = float(np.mean(coords[:, 1]))
    return (center_h, center_w)
