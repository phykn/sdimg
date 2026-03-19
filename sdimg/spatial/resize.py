import cv2
import numpy as np


def resize(
    src: np.ndarray,
    height: int | None = None,
    width: int | None = None,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    dst_h, dst_w = _target_size(
        shape=src.shape[:2],
        height=height,
        width=width,
    )
    return cv2.resize(
        src=src,
        dsize=(dst_w, dst_h),
        interpolation=interpolation,
    )


def resize_keep_ratio(
    src: np.ndarray,
    long_side: int,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    dst_h, dst_w = _long_side_size(
        shape=src.shape[:2],
        long_side=long_side,
    )
    return cv2.resize(
        src=src,
        dsize=(dst_w, dst_h),
        interpolation=interpolation,
    )


def _target_size(
    shape: tuple[int, int],
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    if height is None and width is None:
        raise ValueError("height or width must be provided.")

    src_h, src_w = shape

    if height is not None and height <= 0:
        raise ValueError("height must be greater than 0.")

    if width is not None and width <= 0:
        raise ValueError("width must be greater than 0.")

    if height is None:
        scale = width / src_w
        dst_h = max(1, int(round(src_h * scale)))
        dst_w = width
    elif width is None:
        scale = height / src_h
        dst_h = height
        dst_w = max(1, int(round(src_w * scale)))
    else:
        dst_h = height
        dst_w = width

    return dst_h, dst_w


def _long_side_size(
    shape: tuple[int, int],
    long_side: int,
) -> tuple[int, int]:
    if long_side <= 0:
        raise ValueError("long_side must be greater than 0.")

    src_h, src_w = shape
    src_long_side = max(src_h, src_w)
    scale = long_side / src_long_side

    dst_h = max(1, int(round(src_h * scale)))
    dst_w = max(1, int(round(src_w * scale)))
    return dst_h, dst_w
