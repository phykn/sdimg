import math

import numpy as np

from ..image.helper import to_uint8


def split(
    src: np.ndarray,
    n: int,
    overlap: float = 0.0,
    return_meta: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, object]]:
    data = src

    if n <= 0:
        raise ValueError("n must be greater than 0.")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")

    starts_h, patch_h = _resolve_patch_axis(data.shape[0], n, overlap)
    starts_w, patch_w = _resolve_patch_axis(data.shape[1], n, overlap)

    patches: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []

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

    minimum_size = math.ceil(length / n)

    for patch_size in range(minimum_size, length + 1):
        span = length - patch_size
        starts = np.rint(np.linspace(0, span, num=n)).astype(np.int64).tolist()

        if starts[0] != 0:
            continue
        if starts[-1] + patch_size != length:
            continue

        valid = True
        for left, right in zip(starts, starts[1:]):
            step = right - left
            if step <= 0:
                valid = False
                break
            actual_overlap = 1.0 - (step / patch_size)
            if actual_overlap + 1e-9 < overlap:
                valid = False
                break
        if valid:
            return starts, patch_size

    raise ValueError("Unable to resolve patches for the given n and overlap.")


def _merge_patches(
    patches: list[np.ndarray],
    shape: tuple[int, ...],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    if len(shape) == 2:
        merged = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        weights = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
    else:
        merged = np.zeros(shape, dtype=np.float32)
        weights = np.zeros(shape, dtype=np.float32)

    weight_cache: dict[tuple[int, int], np.ndarray] = {}

    for patch, (wmin, hmin, wmax, hmax) in zip(patches, boxes):
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

    weights = np.maximum(weights, 1e-6)
    merged = merged / weights
    if len(shape) == 2:
        return merged[..., 0]
    return merged
