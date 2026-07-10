import numpy as np
import pytest

from sdimg.image import make_array_id


def test_make_array_id_uses_dtype_shape_and_content() -> None:
    array = np.arange(12, dtype=np.uint16).reshape(3, 4)
    assert make_array_id(array) == make_array_id(array.copy())
    assert make_array_id(array) != make_array_id(array.astype(np.uint8))
    assert make_array_id(array) != make_array_id(array.reshape(2, 6))


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
