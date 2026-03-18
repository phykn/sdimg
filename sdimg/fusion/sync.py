import cv2
import numpy as np

from ..core import is_image, to_mask
from ..spatial import crop, flip, merge, pad, resize, rotate, split, unpad
from ..spatial.patch import SplitMeta


def sync_resize(
    image: np.ndarray,
    mask: np.ndarray,
    height: int | None = None,
    width: int | None = None,
    image_interpolation: int = cv2.INTER_CUBIC,
    mask_interpolation: int = cv2.INTER_NEAREST_EXACT,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_image, normalized_mask = _require_image_mask_pair(image, mask)
    resized_image = resize(
        normalized_image,
        height=height,
        width=width,
        interpolation=image_interpolation,
    )
    resized_mask = resize(
        normalized_mask,
        height=height,
        width=width,
        interpolation=mask_interpolation,
    )
    return resized_image, resized_mask


def sync_rotate(
    image: np.ndarray,
    mask: np.ndarray,
    rotation: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_image, normalized_mask = _require_image_mask_pair(image, mask)
    return rotate(normalized_image, rotation), rotate(normalized_mask, rotation)


def sync_flip(
    image: np.ndarray,
    mask: np.ndarray,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_image, normalized_mask = _require_image_mask_pair(image, mask)
    return flip(normalized_image, direction), flip(normalized_mask, direction)


def sync_pad(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int,
    image_mode: str = "mirror",
    mask_mode: str = "constant",
    mask_value: int = 0,
    return_meta: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    normalized_image, normalized_mask = _require_image_mask_pair(image, mask)
    padded_image = pad(normalized_image, padding, mode=image_mode)
    mask_result = pad(
        normalized_mask,
        padding,
        mode=mask_mode,
        value=mask_value,
        return_meta=return_meta,
    )

    if return_meta:
        padded_mask, meta = mask_result
        return padded_image, padded_mask, meta

    return padded_image, mask_result


def sync_unpad(
    image: np.ndarray,
    mask: np.ndarray,
    pad_width: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    normalized_image, normalized_mask = _require_image_mask_pair(image, mask)
    return unpad(normalized_image, pad_width), unpad(normalized_mask, pad_width)


def sync_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    normalized_image, normalized_mask = _require_image_mask_pair(image, mask)
    return crop(normalized_image, bbox), crop(normalized_mask, bbox)


def sync_split(
    image: np.ndarray,
    mask: np.ndarray,
    n: int,
    overlap: float = 0.0,
    return_meta: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]] | tuple[list[np.ndarray], list[np.ndarray], SplitMeta]:
    normalized_image, normalized_mask = _require_image_mask_pair(image, mask)
    image_result = split(normalized_image, n=n, overlap=overlap, return_meta=return_meta)
    mask_result = split(normalized_mask, n=n, overlap=overlap, return_meta=return_meta)

    if return_meta:
        image_patches, image_meta = image_result
        mask_patches, mask_meta = mask_result
        _require_shared_split_meta(image_meta, mask_meta)
        return image_patches, mask_patches, image_meta

    return image_result, mask_result


def sync_merge(
    image_patches: list[np.ndarray],
    mask_patches: list[np.ndarray],
    meta: SplitMeta,
) -> tuple[np.ndarray, np.ndarray]:
    merged_image = merge(image_patches, dict(meta, kind="image"))
    merged_mask = merge(mask_patches, dict(meta, kind="mask"))
    return merged_image, merged_mask


def _require_image_mask_pair(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not is_image(image):
        raise ValueError("image must be a valid image ndarray.")

    normalized_mask = to_mask(mask)

    if image.shape[:2] != normalized_mask.shape:
        raise ValueError("image and mask must have the same spatial shape.")

    return image.astype(np.uint8, copy=False), normalized_mask


def _require_shared_split_meta(
    image_meta: SplitMeta,
    mask_meta: SplitMeta,
) -> None:
    if image_meta.get("shape")[:2] != mask_meta.get("shape")[:2]:
        raise ValueError("image and mask split metadata must share the same spatial shape.")

    if image_meta.get("boxes") != mask_meta.get("boxes"):
        raise ValueError("image and mask split metadata must share the same boxes.")
