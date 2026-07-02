import numpy as np

from .._core.types import BBox
from .._core.validate import ensure_src
from ..image.convert import to_uint8
from .patch_impl import merge_patches, resolve_patch_axis


def split(
    src: np.ndarray,
    n: int | tuple[int, int],
    overlap: float | tuple[float, float] = 0.0,
    return_meta: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, object]]:
    data = ensure_src(src, name="src")

    if isinstance(n, int):
        nw, nh = n, n
    elif isinstance(n, tuple) and len(n) == 2 and all(isinstance(v, int) for v in n):
        nw, nh = n
    else:
        raise TypeError("n must be an int or a tuple of two ints.")

    if nw <= 0 or nh <= 0:
        raise ValueError("n must be greater than 0.")

    if isinstance(overlap, (int, float)):
        overlap_w, overlap_h = float(overlap), float(overlap)
    elif (
        isinstance(overlap, tuple)
        and len(overlap) == 2
        and all(isinstance(v, (int, float)) for v in overlap)
    ):
        overlap_w, overlap_h = float(overlap[0]), float(overlap[1])
    else:
        raise TypeError("overlap must be a float or a tuple of two floats.")

    if not (0.0 <= overlap_w < 1.0) or not (0.0 <= overlap_h < 1.0):
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")

    starts_h, patch_h = resolve_patch_axis(data.shape[0], nh, overlap_h)
    starts_w, patch_w = resolve_patch_axis(data.shape[1], nw, overlap_w)

    patches: list[np.ndarray] = []
    boxes: list[BBox] = []

    for hmin in starts_h:
        hmax = hmin + patch_h
        for wmin in starts_w:
            wmax = wmin + patch_w
            patches.append(data[hmin:hmax, wmin:wmax].copy())
            boxes.append((wmin, hmin, wmax, hmax))

    if not return_meta:
        return patches

    meta: dict[str, object] = {
        "shape": data.shape,
        "boxes": boxes,
    }
    return patches, meta


def merge(
    patches: list[np.ndarray],
    meta: dict[str, object],
) -> np.ndarray:
    if len(patches) == 0:
        raise ValueError("patches must not be empty.")

    if "shape" not in meta or "boxes" not in meta:
        raise ValueError("meta must include shape and boxes.")
    shape = meta["shape"]
    boxes = meta["boxes"]

    if not isinstance(shape, tuple):
        raise ValueError("meta['shape'] must be a tuple.")
    if len(shape) not in {2, 3}:
        raise ValueError("meta['shape'] must have length 2 or 3.")
    if not isinstance(boxes, list):
        raise ValueError("meta['boxes'] must be a list.")

    if len(patches) != len(boxes):
        raise ValueError("patches and meta boxes length must match.")

    merged = merge_patches(
        patches,
        tuple(shape),
        list(boxes),
    )
    return to_uint8(merged)
