import hashlib

import numpy as np

from ..core.validation import validate_array


def make_array_id(
    array: np.ndarray,
    *,
    prefix: str = "",
    length: int = 8,
) -> str:
    array = validate_array(array)
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a str.")
    if not isinstance(length, int) or isinstance(length, bool):
        raise TypeError("length must be an int.")
    if not 1 <= length <= 32:
        raise ValueError("length must be between 1 and 32.")
    if array.dtype.hasobject:
        raise ValueError("array must not have object dtype.")

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.md5()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous))
    return prefix + digest.hexdigest()[:length]
