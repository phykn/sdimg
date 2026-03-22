import numpy as np

from .helper import get_bbox


def to_roi(
    mask: np.ndarray,
) -> dict[str, object] | None:
    """Extract the bounding region of a mask as a ROI.

    Returns:
        dict with 'roi' (cropped binary mask copy) and
        'box' (wmin, hmin, wmax, hmax), or None if the mask is empty.
    """
    bbox = get_bbox(mask)
    if bbox is None:
        return None

    wmin, hmin, wmax, hmax = bbox
    roi = mask[hmin:hmax, wmin:wmax].copy()
    return {"roi": roi, "box": bbox}
