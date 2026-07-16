import cv2
import numpy as np

from ..core.validation import validate_positive_int, validate_source
from .dtype import restore_dtype

_CV2_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.float32),
    np.dtype(np.float64),
}


def resize(
    array: np.ndarray,
    height: int | None = None,
    width: int | None = None,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    array = validate_source(array)
    height = _validate_optional_size(height, "height")
    width = _validate_optional_size(width, "width")
    if height is None and width is None:
        raise ValueError("height or width must be provided.")
    destination = _resolve_size(array.shape[:2], height, width)
    return _resize(array, destination, interpolation, "resize")


def resize_to_long_side(
    array: np.ndarray,
    long_side: int,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    array = validate_source(array)
    long_side = validate_positive_int(long_side, "long_side")
    source_height, source_width = array.shape[:2]
    scale = long_side / max(source_height, source_width)
    destination = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )
    return _resize(array, destination, interpolation, "resize_to_long_side")


def _resolve_size(
    shape: tuple[int, int],
    height: int | None,
    width: int | None,
) -> tuple[int, int]:
    source_height, source_width = shape
    if height is None:
        assert width is not None
        height = max(1, int(round(source_height * (width / source_width))))
    elif width is None:
        width = max(1, int(round(source_width * (height / source_height))))
    return (width, height)


def _resize(
    array: np.ndarray,
    destination: tuple[int, int],
    interpolation: int,
    function_name: str,
) -> np.ndarray:
    if not isinstance(interpolation, int) or isinstance(interpolation, bool):
        raise TypeError("interpolation must be an int.")
    original_dtype = array.dtype
    working = array if original_dtype in _CV2_DTYPES else array.astype(np.float64)
    try:
        result = cv2.resize(working, destination, interpolation=interpolation)
    except Exception as exc:
        raise RuntimeError(f"{function_name} failed: {exc}") from exc
    if array.ndim == 3 and array.shape[2] == 1 and result.ndim == 2:
        result = result[..., np.newaxis]
    return np.ascontiguousarray(restore_dtype(result, original_dtype))


def _validate_optional_size(value: object, name: str) -> int | None:
    if value is None:
        return None
    return validate_positive_int(value, name)
