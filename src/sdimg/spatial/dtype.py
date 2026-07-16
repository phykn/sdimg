import numpy as np


_FLOAT64_EXACT_INTEGER_LIMIT = 1 << 53


def validate_float64_integer_range(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.integer):
        return
    minimum = array.min().item()
    maximum = array.max().item()
    if (
        minimum < -_FLOAT64_EXACT_INTEGER_LIMIT
        or maximum > _FLOAT64_EXACT_INTEGER_LIMIT
    ):
        raise ValueError(
            f"{name} integer values must be within "
            f"[-{_FLOAT64_EXACT_INTEGER_LIMIT}, {_FLOAT64_EXACT_INTEGER_LIMIT}] "
            "for float64-backed spatial operations."
        )


def restore_dtype(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.bool_:
        return array >= 0.5
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.rint(np.clip(array, limits.min, limits.max)).astype(dtype)
    return array.astype(dtype, copy=False)
