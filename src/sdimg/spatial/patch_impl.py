import math

import numpy as np

from ..core.types import BBox


def resolve_patch_axis(
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


def merge_patches(
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

        patch_weights = _patch_weights(validated.shape[:2], weight_cache)
        merged[hmin:hmax, wmin:wmax, :] += validated * patch_weights
        weights[hmin:hmax, wmin:wmax, :] += patch_weights

    np.maximum(weights, 1e-6, out=weights)
    merged /= weights
    if len(shape) == 2:
        return merged[..., 0]
    return merged


def _patch_weights(
    patch_shape: tuple[int, int],
    cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    patch_weights = cache.get(patch_shape)
    if patch_weights is None:
        h, w = patch_shape
        patch_weights = _axis_weights(h)[:, None] * _axis_weights(w)[None, :]
        cache[patch_shape] = patch_weights
    return patch_weights[..., None]


def _axis_weights(length: int) -> np.ndarray:
    if length == 1:
        return np.ones(1, dtype=np.float32)

    axis = np.linspace(0.0, np.pi, num=length, dtype=np.float32)
    return np.maximum(0.5 - 0.5 * np.cos(axis), 1e-3)
