import math

import numpy as np

from ..core import is_mask
from ._common import _finalize_image, _finalize_mask, _require_spatial_input


type SplitMeta = dict[str, object]


def split(
    src: np.ndarray,
    n: int,
    overlap: float = 0.0,
    return_meta: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], SplitMeta]:
    data = _require_spatial_input(src)
    _require_split_args(n, overlap)

    starts_h, patch_height = _resolve_patch_axis(data.shape[0], n, overlap)
    starts_w, patch_width = _resolve_patch_axis(data.shape[1], n, overlap)

    patches: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []

    for hmin in starts_h:
        hmax = hmin + patch_height
        for wmin in starts_w:
            wmax = wmin + patch_width
            patches.append(data[hmin:hmax, wmin:wmax].copy())
            boxes.append((wmin, hmin, wmax, hmax))

    if not return_meta:
        return patches

    meta: SplitMeta = {
        "shape": data.shape,
        "boxes": boxes,
        "kind": "mask" if is_mask(data) else "image",
    }
    return patches, meta


def merge(
    patches: list[np.ndarray],
    meta: SplitMeta,
) -> np.ndarray:
    if len(patches) == 0:
        raise ValueError("patches must not be empty.")

    shape = _require_meta_shape(meta)
    boxes = _require_meta_boxes(meta)

    if len(patches) != len(boxes):
        raise ValueError("patches and meta boxes length must match.")

    if _require_meta_kind(meta) == "mask":
        merged = _merge_mask_patches(patches, shape, boxes)
        return _finalize_mask(merged)

    merged = _merge_image_patches(patches, shape, boxes)
    return _finalize_image(merged)


def _require_split_args(
    n: int,
    overlap: float,
) -> None:
    if n <= 0:
        raise ValueError("n must be greater than 0.")

    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")


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

        if _is_valid_patch_axis(starts, patch_size, length, overlap):
            return starts, patch_size

    raise ValueError("Unable to resolve patches for the given n and overlap.")


def _is_valid_patch_axis(
    starts: list[int],
    patch_size: int,
    length: int,
    overlap: float,
) -> bool:
    if starts[0] != 0:
        return False

    if starts[-1] + patch_size != length:
        return False

    for left, right in zip(starts, starts[1:]):
        step = right - left
        if step <= 0:
            return False

        actual_overlap = 1.0 - (step / patch_size)
        if actual_overlap + 1e-9 < overlap:
            return False

    return True


def _require_meta_shape(meta: SplitMeta) -> tuple[int, ...]:
    shape = meta.get("shape")
    if not isinstance(shape, tuple) or len(shape) not in {2, 3}:
        raise ValueError("meta shape is invalid.")
    return shape


def _require_meta_boxes(
    meta: SplitMeta,
) -> list[tuple[int, int, int, int]]:
    boxes = meta.get("boxes")
    if not isinstance(boxes, list) or len(boxes) == 0:
        raise ValueError("meta boxes are invalid.")

    validated: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not isinstance(box, tuple) or len(box) != 4:
            raise ValueError("meta boxes are invalid.")
        validated.append(box)
    return validated


def _require_meta_kind(meta: SplitMeta) -> str:
    kind = meta.get("kind")
    if kind not in {"image", "mask"}:
        raise ValueError("meta kind is invalid.")
    return kind


def _merge_mask_patches(
    patches: list[np.ndarray],
    shape: tuple[int, ...],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    merged = np.zeros(shape[:2], dtype=np.uint8)

    for patch, (wmin, hmin, wmax, hmax) in zip(patches, boxes):
        normalized = _finalize_mask(_require_patch_shape(patch, hmax - hmin, wmax - wmin))
        merged[hmin:hmax, wmin:wmax] |= normalized

    return merged


def _merge_image_patches(
    patches: list[np.ndarray],
    shape: tuple[int, ...],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    merged = np.zeros(shape, dtype=np.float32)
    weights = np.zeros(shape, dtype=np.float32)

    for patch, (wmin, hmin, wmax, hmax) in zip(patches, boxes):
        validated = _require_patch_shape(patch, hmax - hmin, wmax - wmin).astype(np.float32, copy=False)
        patch_weights = _build_patch_weights(validated.shape[:2])

        if validated.ndim == 2:
            merged[hmin:hmax, wmin:wmax] += validated * patch_weights
            weights[hmin:hmax, wmin:wmax] += patch_weights
            continue

        merged[hmin:hmax, wmin:wmax, :] += validated * patch_weights[..., None]
        weights[hmin:hmax, wmin:wmax, :] += patch_weights[..., None]

    weights = np.maximum(weights, 1e-6)
    return merged / weights


def _require_patch_shape(
    patch: np.ndarray,
    expected_height: int,
    expected_width: int,
) -> np.ndarray:
    if not isinstance(patch, np.ndarray):
        raise ValueError("Each patch must be a numpy.ndarray.")

    if patch.shape[:2] != (expected_height, expected_width):
        raise ValueError("Patch shape does not match meta boxes.")

    return patch


def _build_patch_weights(
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    return _build_axis_weights(height)[:, None] * _build_axis_weights(width)[None, :]


def _build_axis_weights(length: int) -> np.ndarray:
    if length == 1:
        return np.ones(1, dtype=np.float32)

    axis = np.linspace(0.0, np.pi, num=length, dtype=np.float32)
    weights = 0.5 - 0.5 * np.cos(axis)
    return np.maximum(weights, 1e-3)
