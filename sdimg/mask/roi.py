import numpy as np

from .helper import get_coords


def to_roi(
    mask: np.ndarray,
) -> dict[str, object] | None:
    coords = get_coords(mask)
    if coords.shape[0] == 0:
        return None

    hmin = int(coords[:, 0].min())
    hmax = int(coords[:, 0].max()) + 1
    wmin = int(coords[:, 1].min())
    wmax = int(coords[:, 1].max()) + 1

    roi = mask[hmin:hmax, wmin:wmax]
    return {"roi": roi, "box": (wmin, hmin, wmax, hmax)}
