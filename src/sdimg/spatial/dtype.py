import numpy as np


def restore_dtype(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.bool_:
        return array >= 0.5
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.rint(np.clip(array, limits.min, limits.max)).astype(dtype)
    return array.astype(dtype, copy=False)
