import hashlib
import numpy as np
import pytest

from sdimg.image import make_array_id


def test_make_array_id_uses_dtype_shape_and_content() -> None:
    array = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert make_array_id(array) == make_array_id(array.copy())
    assert make_array_id(array) != make_array_id(array.astype(np.uint8))
    assert make_array_id(array) != make_array_id(array.reshape(2, 6))


def test_make_array_id_distinguishes_zero_one_and_two_dimensional_shapes() -> None:
    arrays = [
        np.array(1, dtype=np.uint8),
        np.array([1], dtype=np.uint8),
        np.array([[1]], dtype=np.uint8),
    ]

    identifiers = {make_array_id(array, length=32) for array in arrays}

    assert len(identifiers) == len(arrays)


def test_make_array_id_supports_prefix_length_and_non_contiguous_array() -> None:
    array = np.arange(16, dtype=np.uint8).reshape(4, 4)[::2]
    value = make_array_id(array, prefix="img_", length=16)
    assert value.startswith("img_")
    assert len(value) == 20
    assert value == make_array_id(np.ascontiguousarray(array), prefix="img_", length=16)


@pytest.mark.parametrize("prefix", [1, None, b"img_"])
def test_make_array_id_rejects_non_string_prefix(prefix: object) -> None:
    with pytest.raises(TypeError, match="prefix"):
        make_array_id(np.zeros((2, 2), dtype=np.uint8), prefix=prefix)  # type: ignore[arg-type]


def test_make_array_id_rejects_object_dtype() -> None:
    with pytest.raises(ValueError, match="object"):
        make_array_id(np.array([object()], dtype=object))


def test_make_array_id_preserves_plain_dtype_identifier() -> None:
    array = np.arange(4, dtype=np.uint8)

    legacy = hashlib.md5()
    legacy.update(array.dtype.str.encode("ascii"))
    legacy.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    legacy.update(memoryview(array))
    assert make_array_id(array, length=32) == legacy.hexdigest()


def test_make_array_id_distinguishes_structured_field_names_and_layout() -> None:
    named_left = np.zeros(2, dtype=[("left", "u1"), ("right", "u1")])
    named_xy = np.zeros(2, dtype=[("x", "u1"), ("y", "u1")])
    offset_zero = np.zeros(
        2,
        dtype=np.dtype(
            {"names": ["value"], "formats": ["u1"], "offsets": [0], "itemsize": 2}
        ),
    )
    offset_one = np.zeros(
        2,
        dtype=np.dtype(
            {"names": ["value"], "formats": ["u1"], "offsets": [1], "itemsize": 2}
        ),
    )

    assert make_array_id(named_left, length=32) != make_array_id(
        named_xy, length=32
    )
    assert make_array_id(offset_zero, length=32) != make_array_id(
        offset_one, length=32
    )


def test_make_array_id_is_stable_for_noncontiguous_structured_array() -> None:
    array = np.zeros(4, dtype=[("left", "u1"), ("right", "u1")])[::2]

    assert make_array_id(array, length=32) == make_array_id(
        np.ascontiguousarray(array), length=32
    )
