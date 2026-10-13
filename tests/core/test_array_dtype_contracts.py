from collections.abc import Callable

import numpy as np
import pytest

from sdimg.core.validation import validate_image, validate_source
from sdimg.image import convert_to_uint8, is_image, make_array_id
from sdimg.mask import convert_to_mask, is_mask
from sdimg.spatial import resize


@pytest.mark.parametrize("dtype", ["datetime64[D]", "timedelta64[D]", "timedelta64[ns]"])
@pytest.mark.parametrize(
    "function", [validate_source, validate_image, convert_to_uint8, convert_to_mask]
)
def test_processing_rejects_temporal_dtypes(
    function: Callable[[np.ndarray], np.ndarray], dtype: str
) -> None:
    array = np.array([[0, 1]], dtype=dtype)

    with pytest.raises(ValueError, match="numeric or boolean dtype"):
        function(array)


@pytest.mark.parametrize("dtype", ["datetime64[D]", "timedelta64[D]", "timedelta64[ns]"])
def test_predicates_reject_temporal_dtypes(dtype: str) -> None:
    array = np.array([[0, 1]], dtype=dtype)

    assert not is_image(array)
    assert not is_mask(array)


def test_resize_rejects_duration_values_at_the_input_boundary() -> None:
    array = np.array([[0, 1]], dtype="timedelta64[D]")

    with pytest.raises(ValueError, match="numeric or boolean dtype"):
        resize(array, width=4)


def test_array_identity_still_accepts_temporal_dtypes() -> None:
    array = np.array([[0, 1]], dtype="timedelta64[D]")

    assert make_array_id(array) == make_array_id(array.copy())
    assert make_array_id(array) != make_array_id(array.astype("timedelta64[ns]"))
