import math

import numpy as np

from .._core.types import BBox
from .._core.validate import ensure_src
from ..image.convert import to_uint8


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

    starts_h, patch_h = _resolve_patch_axis(data.shape[0], nh, overlap_h)
    starts_w, patch_w = _resolve_patch_axis(data.shape[1], nw, overlap_w)

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

    merged = _merge_patches(
        patches,
        tuple(shape),
        list(boxes),
    )
    return to_uint8(merged)


def _resolve_patch_axis(
    length: int,
    n: int,
    overlap: float,
) -> tuple[list[int], int]:
    if n == 1:
        return [0], length

    # closed-form lower bound: step = (length - P) / (n - 1), overlap = 1 - step/P
    denominator = 1.0 + (n - 1) * (1.0 - overlap)
    patch_size = max(1, math.ceil(length / denominator))

    while patch_size <= length:
        span = length - patch_size
        starts = np.rint(np.linspace(0, span, num=n)).astype(np.int64).tolist()

        valid = True
        for left, right in zip(starts, starts[1:]):
            step = right - left
            if step <= 0:
                valid = False
                break
            if 1.0 - (step / patch_size) + 1e-9 < overlap:
                valid = False
                break
        if valid:
            return starts, patch_size
        patch_size += 1

    raise ValueError("Unable to resolve patches for the given n and overlap.")


def _merge_patches(
    patches: list[np.ndarray],
    shape: tuple[int, ...],
    boxes: list[BBox],
) -> np.ndarray:
    if len(shape) == 2:
        merged = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        weights = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
    else:
        merged = np.zeros(shape, dtype=np.float32)
        weights = np.zeros(shape, dtype=np.float32)

    weight_cache: dict[tuple[int, int], np.ndarray] = {}

    for patch, (wmin, hmin, wmax, hmax) in zip(patches, boxes):
        if not all(isinstance(v, int) for v in (wmin, hmin, wmax, hmax)):
            raise ValueError("Each box must be a tuple of 4 integers.")
        if wmin < 0 or hmin < 0 or wmax > shape[1] or hmax > shape[0]:
            raise ValueError("Each box must be within output shape bounds.")
        if wmin >= wmax or hmin >= hmax:
            raise ValueError("Each box must satisfy wmin < wmax and hmin < hmax.")

        if not isinstance(patch, np.ndarray):
            raise ValueError("Each patch must be a numpy.ndarray.")
        if patch.shape[:2] != (hmax - hmin, wmax - wmin):
            raise ValueError("Patch shape does not match meta boxes.")
        validated = patch.astype(np.float32, copy=False)
        if validated.ndim == 2:
            validated = validated[..., None]

        patch_shape = validated.shape[:2]
        patch_weights = weight_cache.get(patch_shape)
        if patch_weights is None:
            h, w = patch_shape
            if h == 1:
                weights_h = np.ones(1, dtype=np.float32)
            else:
                axis_h = np.linspace(0.0, np.pi, num=h, dtype=np.float32)
                weights_h = np.maximum(0.5 - 0.5 * np.cos(axis_h), 1e-3)

            if w == 1:
                weights_w = np.ones(1, dtype=np.float32)
            else:
                axis_w = np.linspace(0.0, np.pi, num=w, dtype=np.float32)
                weights_w = np.maximum(0.5 - 0.5 * np.cos(axis_w), 1e-3)

            patch_weights = weights_h[:, None] * weights_w[None, :]
            weight_cache[patch_shape] = patch_weights

        patch_weights = patch_weights[..., None]
        merged[hmin:hmax, wmin:wmax, :] += validated * patch_weights
        weights[hmin:hmax, wmin:wmax, :] += patch_weights

    np.maximum(weights, 1e-6, out=weights)
    merged /= weights
    if len(shape) == 2:
        return merged[..., 0]
    return merged
