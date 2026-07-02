import numpy as np

from ..core.validate import ensure_mask


def is_mask(mask: object) -> bool:
    try:
        to_mask(mask)
    except (TypeError, ValueError):
        return False

    return True


def to_mask(mask: object) -> np.ndarray:
    mask = ensure_mask(mask, name="Mask input")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    if mask.size == 0:
        return mask.astype(np.uint8)

    mx = mask.max()
    mn = mask.min()

    if mn >= 0 and mx <= 1:
        if np.issubdtype(mask.dtype, np.floating):
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(
                    "Mask input must contain only binary values represented as "
                    "bool, {0, 1}, or {0, 255}.",
                )
        return mask.astype(np.uint8)

    if mx == 255 and mn >= 0:
        if bool(((mask == 0) | (mask == 255)).all()):
            return (mask > 0).astype(np.uint8)

    raise ValueError(
        "Mask input must contain only binary values represented as "
        "bool, {0, 1}, or {0, 255}.",
    )
