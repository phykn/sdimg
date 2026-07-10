import numpy as np
import pytest

from sdimg.image import make_array_id


def test_make_array_id_is_deterministic() -> None:
    arr = np.zeros((4, 4), dtype=np.uint8)

    assert make_array_id(arr) == make_array_id(arr)


def test_make_array_id_changes_with_content_shape_and_dtype() -> None:
    base = np.zeros((4, 4), dtype=np.uint8)

    assert make_array_id(base) != make_array_id(np.ones((4, 4), dtype=np.uint8))
    assert make_array_id(base) != make_array_id(np.zeros((2, 8), dtype=np.uint8))
    assert make_array_id(base) != make_array_id(np.zeros((4, 4), dtype=np.float32))


def test_make_array_id_handles_prefix_length_and_non_contiguous_arrays() -> None:
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    sliced = arr[::2]

    assert make_array_id(arr, prefix="img_").startswith("img_")
    assert len(make_array_id(arr, length=16)) == 16
    assert make_array_id(sliced) == make_array_id(np.ascontiguousarray(sliced))


def test_make_array_id_rejects_non_ndarray_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        make_array_id([1, 2, 3])  # type: ignore[arg-type]


def test_make_array_id_rejects_object_dtype() -> None:
    arr = np.array(["alpha", "beta"], dtype=object)

    with pytest.raises(ValueError, match="object dtype"):
        make_array_id(arr)


@pytest.mark.parametrize("length", [0, -1, 33])
def test_make_array_id_rejects_invalid_length_value(length: int) -> None:
    with pytest.raises(ValueError, match="length must be between 1 and 32"):
        make_array_id(np.zeros((2, 2), dtype=np.uint8), length=length)


@pytest.mark.parametrize("length", [True, 1.0, "8"])
def test_make_array_id_rejects_invalid_length_type(length: object) -> None:
    with pytest.raises(TypeError, match="length must be an int"):
        make_array_id(np.zeros((2, 2), dtype=np.uint8), length=length)  # type: ignore[arg-type]
