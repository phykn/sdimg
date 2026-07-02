import hashlib

import numpy as np

from ..core.validate import ensure_ndarray


def get_id(arr: np.ndarray, *, prefix: str = "", length: int = 8) -> str:
    arr = ensure_ndarray(arr, name="arr")
    if not isinstance(length, int) or isinstance(length, bool):
        raise TypeError("length must be an int.")
    if not 1 <= length <= 32:
        raise ValueError("length must be between 1 and 32.")
    if arr.dtype.hasobject:
        raise ValueError("arr must not have object dtype.")

    contiguous_arr = np.ascontiguousarray(arr)
    hasher = hashlib.md5()
    hasher.update(contiguous_arr.dtype.str.encode("ascii"))
    hasher.update(np.asarray(contiguous_arr.shape, dtype=np.int64).tobytes())
    hasher.update(memoryview(contiguous_arr))
    return prefix + hasher.hexdigest()[:length]
