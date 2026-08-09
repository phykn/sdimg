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

    shape = array.shape
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.md5()
    digest.update(_encode_dtype(contiguous.dtype))
    digest.update(np.asarray(shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous))
    return prefix + digest.hexdigest()[:length]


def _encode_dtype(dtype: np.dtype) -> bytes:
    if dtype.fields is None:
        return dtype.str.encode("ascii")
    return ("structured:" + repr(_dtype_signature(dtype))).encode("utf-8")


def _dtype_signature(dtype: np.dtype) -> tuple[object, ...] | str:
    if dtype.subdtype is not None:
        base, shape = dtype.subdtype
        return ("subarray", _dtype_signature(base), tuple(shape))
    if dtype.fields is None:
        return dtype.str

    fields: list[tuple[object, ...]] = []
    for name in dtype.names or ():
        field = dtype.fields[name]
        field_dtype, offset = field[:2]
        title = field[2] if len(field) == 3 else None
        fields.append((name, int(offset), title, _dtype_signature(field_dtype)))
    return (
        "structured",
        dtype.itemsize,
        dtype.alignment,
        dtype.isalignedstruct,
        tuple(fields),
    )
