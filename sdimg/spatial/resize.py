import cv2
import numpy as np

from ..core import is_mask
from ._common import _finalize_image, _finalize_mask, _require_spatial_input


def resize(
    src: np.ndarray,
    height: int | None = None,
    width: int | None = None,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    data = _require_spatial_input(src)
    target_height, target_width = _resolve_target_size(data, height, width)

    resized = cv2.resize(
        data,
        (target_width, target_height),
        interpolation=interpolation,
    )

    if is_mask(data):
        return _finalize_mask(resized)

    return _finalize_image(resized)


def _resolve_target_size(
    src: np.ndarray,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    if height is None and width is None:
        raise ValueError("height or width must be provided.")

    source_height, source_width = src.shape[:2]

    if height is not None and height <= 0:
        raise ValueError("height must be greater than 0.")

    if width is not None and width <= 0:
        raise ValueError("width must be greater than 0.")

    if height is not None and width is not None:
        return (height, width)

    if height is not None:
        scale = height / source_height
        resolved_width = max(1, int(round(source_width * scale)))
        return (height, resolved_width)

    assert width is not None
    scale = width / source_width
    resolved_height = max(1, int(round(source_height * scale)))
    return (resolved_height, width)
