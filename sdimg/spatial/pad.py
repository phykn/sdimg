import numpy as np

from ..core import is_mask
from ._common import _finalize_image, _finalize_mask, _require_spatial_input


def pad(
    src: np.ndarray,
    padding: int,
    mode: str | None = None,
    value: int = 0,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int, int]]:
    data = _require_spatial_input(src)
    pad_width = _resolve_pad_width(padding, data.ndim)
    pad_mode = _resolve_pad_mode(data, mode)
    padded = _apply_padding(data, pad_width, pad_mode, value)
    result = _finalize_spatial_output(data, padded)
    meta = (padding, padding, padding, padding)

    if return_meta:
        return result, meta

    return result


def unpad(
    src: np.ndarray,
    pad_width: tuple[int, int, int, int],
) -> np.ndarray:
    data = _require_spatial_input(src)
    top, bottom, left, right = _require_pad_width(pad_width)
    height, width = data.shape[:2]

    if top + bottom > height or left + right > width:
        raise ValueError("pad_width exceeds input shape.")

    hmin = top
    hmax = height - bottom
    wmin = left
    wmax = width - right
    unpadded = data[hmin:hmax, wmin:wmax]
    return _finalize_spatial_output(data, unpadded)


def _resolve_pad_width(
    padding: int,
    ndim: int,
) -> tuple[tuple[int, int], ...]:
    if padding < 0:
        raise ValueError("padding must be greater than or equal to 0.")

    spatial_pad = ((padding, padding), (padding, padding))
    if ndim == 2:
        return spatial_pad

    return spatial_pad + ((0, 0),)


def _resolve_pad_mode(
    src: np.ndarray,
    mode: str | None,
) -> str:
    if mode is None:
        if is_mask(src):
            return "constant"
        return "mirror"

    if mode not in {"constant", "reflect", "mirror", "edge"}:
        raise ValueError("mode must be one of constant, reflect, mirror, edge.")

    return mode


def _apply_padding(
    src: np.ndarray,
    pad_width: tuple[tuple[int, int], ...],
    mode: str,
    value: int,
) -> np.ndarray:
    if mode == "constant":
        return np.pad(
            src,
            pad_width,
            mode="constant",
            constant_values=value,
        )

    if mode == "reflect":
        return np.pad(src, pad_width, mode="reflect")

    if mode == "mirror":
        return np.pad(src, pad_width, mode="symmetric")

    return np.pad(src, pad_width, mode="edge")


def _require_pad_width(
    pad_width: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if len(pad_width) != 4:
        raise ValueError("pad_width must be a 4-tuple of (top, bottom, left, right).")

    top, bottom, left, right = pad_width
    values = (top, bottom, left, right)

    if any(not isinstance(value, int) for value in values):
        raise ValueError("pad_width values must be integers.")

    if any(value < 0 for value in values):
        raise ValueError("pad_width values must be greater than or equal to 0.")

    return values


def _finalize_spatial_output(
    original: np.ndarray,
    result: np.ndarray,
) -> np.ndarray:
    if is_mask(original):
        return _finalize_mask(result)

    return _finalize_image(result)
