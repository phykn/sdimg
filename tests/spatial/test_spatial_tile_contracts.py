import itertools
import math
import warnings

import numpy as np
import pytest

from sdimg.spatial import merge_tiles, split_tiles


def test_merge_tiles_averages_overlap_without_direction_bias() -> None:
    tiles = [
        np.zeros((3, 3), dtype=np.uint8),
        np.full((3, 3), 100, dtype=np.uint8),
    ]
    metadata = {"shape": (3, 5), "boxes": [(0, 0, 3, 3), (2, 0, 5, 3)]}
    out = merge_tiles(tiles, metadata)
    assert np.all(out[:, 2] == 50)


def test_merge_tiles_is_order_independent() -> None:
    tiles = [
        np.zeros((3, 3), dtype=np.uint8),
        np.full((3, 3), 100, dtype=np.uint8),
    ]
    boxes = [(0, 0, 3, 3), (2, 0, 5, 3)]
    forward = merge_tiles(tiles, {"shape": (3, 5), "boxes": boxes})
    reverse = merge_tiles(tiles[::-1], {"shape": (3, 5), "boxes": boxes[::-1]})
    assert np.array_equal(forward, reverse)


def test_split_merge_float64_max_does_not_overflow() -> None:
    array = np.full((5, 5), np.finfo(np.float64).max)
    tiles, metadata = split_tiles(
        array,
        grid=(2, 2),
        overlap=0.5,
        return_metadata=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = merge_tiles(tiles, metadata)

    assert np.array_equal(out, array)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_split_merge_preserves_nonfinite_float_values(
    dtype: np.dtype,
    value: float,
) -> None:
    array = np.full((2, 2), value, dtype=dtype)
    tiles, metadata = split_tiles(array, grid=1, return_metadata=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = merge_tiles(tiles, metadata)

    assert out.dtype == array.dtype
    assert np.array_equal(out, array, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "values,expected",
    [
        ([np.inf, 1.0], np.inf),
        ([-np.inf, 1.0], -np.inf),
        ([np.inf, -np.inf], np.nan),
        ([np.nan, np.inf], np.nan),
    ],
)
def test_merge_tiles_combines_nonfinite_values_without_warnings(
    dtype: np.dtype,
    values: list[float],
    expected: float,
) -> None:
    tiles = [np.full((2, 2), value, dtype=dtype) for value in values]
    metadata = {
        "shape": (2, 2),
        "boxes": [(0, 0, 2, 2)] * len(tiles),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = merge_tiles(tiles, metadata)

    assert out.dtype == dtype
    if np.isnan(expected):
        assert np.isnan(out).all()
    else:
        assert np.all(out == expected)


def test_merge_tiles_averages_opposite_float64_extremes_without_overflow() -> None:
    limit = np.finfo(np.float64).max
    tiles = [np.full((2, 2), limit), np.full((2, 2), -limit)]
    metadata = {"shape": (2, 2), "boxes": [(0, 0, 2, 2), (0, 0, 2, 2)]}

    for ordered_tiles in (tiles, tiles[::-1]):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = merge_tiles(ordered_tiles, metadata)

        assert np.array_equal(out, np.zeros((2, 2), dtype=np.float64))


@pytest.mark.parametrize(
    "values,dtype,expected",
    [
        ([100, 2**53 - 1, -(2**53)], np.int64, 33),
        ([4, 18, 212, 216], np.uint8, 112),
    ],
)
def test_merge_tiles_integer_average_is_exact_and_order_independent(
    values: list[int],
    dtype: np.dtype,
    expected: int,
) -> None:
    metadata = {
        "shape": (1, 1),
        "boxes": [(0, 0, 1, 1)] * len(values),
    }

    for permutation in itertools.permutations(values):
        tiles = [np.array([[value]], dtype=dtype) for value in permutation]
        out = merge_tiles(tiles, metadata)

        assert out.dtype == dtype
        assert out.item() == expected


def test_merge_tiles_float_average_is_order_independent() -> None:
    values = [2**-53, 1.0, 2**-53, float(2**53)]
    expected = math.fsum(values) / len(values)
    metadata = {
        "shape": (1, 1),
        "boxes": [(0, 0, 1, 1)] * len(values),
    }

    for permutation in set(itertools.permutations(values)):
        tiles = [np.array([[value]], dtype=np.float64) for value in permutation]
        out = merge_tiles(tiles, metadata)

        assert out.item() == expected


@pytest.mark.parametrize(
    "array",
    [
        np.arange(80, dtype=np.float32).reshape(8, 10),
        np.arange(80, dtype=np.int16).reshape(8, 10),
        np.arange(80).reshape(8, 10) % 3 == 0,
    ],
)
def test_split_merge_roundtrip_preserves_dtype_and_values(array: np.ndarray) -> None:
    tiles, metadata = split_tiles(
        array,
        grid=2,
        overlap=0.25,
        return_metadata=True,
    )
    out = merge_tiles(tiles, metadata)
    assert out.dtype == array.dtype
    assert np.array_equal(out, array)


@pytest.mark.parametrize(
    "array",
    [
        np.array([[-(2**53), 2**53]], dtype=np.int64),
        np.array([[0, 2**53]], dtype=np.uint64),
    ],
)
def test_split_merge_roundtrip_accepts_float64_exact_integer_boundary(
    array: np.ndarray,
) -> None:
    tiles, metadata = split_tiles(array, grid=1, return_metadata=True)

    out = merge_tiles(tiles, metadata)

    assert out.dtype == array.dtype
    assert np.array_equal(out, array)


@pytest.mark.parametrize(
    "array",
    [
        np.array([[-(2**53) - 1]], dtype=np.int64),
        np.array([[2**53 + 1]], dtype=np.int64),
        np.array([[2**53 + 1]], dtype=np.uint64),
    ],
)
def test_merge_tiles_rejects_integers_outside_float64_exact_range(
    array: np.ndarray,
) -> None:
    tiles, metadata = split_tiles(array, grid=1, return_metadata=True)

    with pytest.raises(ValueError, match="float64-backed spatial operations"):
        merge_tiles(tiles, metadata)


def test_split_tiles_supports_asymmetric_grid_and_overlap() -> None:
    array = np.zeros((10, 20, 3), dtype=np.uint8)
    tiles, metadata = split_tiles(
        array,
        grid=(3, 2),
        overlap=(0.5, 0.0),
        return_metadata=True,
    )
    assert len(tiles) == 6
    assert metadata["boxes"][0][:2] == (0, 0)
    assert metadata["boxes"][-1][2:] == (20, 10)


def test_split_tiles_meets_high_overlap_and_covers_both_ends() -> None:
    array = np.zeros((100, 100), dtype=np.uint8)
    tiles, metadata = split_tiles(
        array,
        grid=4,
        overlap=0.5,
        return_metadata=True,
    )
    assert len(tiles) == 16
    boxes = metadata["boxes"]
    assert boxes[0][:2] == (0, 0)
    assert boxes[-1][2:] == (100, 100)
    for left, right in zip(boxes[:3], boxes[1:4]):
        width = left[2] - left[0]
        step = right[0] - left[0]
        assert 1.0 - step / width + 1e-9 >= 0.5


def test_split_tiles_roundtrips_non_divisible_shape() -> None:
    array = np.arange(17 * 23, dtype=np.uint8).reshape(17, 23)
    tiles, metadata = split_tiles(
        array,
        grid=3,
        overlap=0.3,
        return_metadata=True,
    )
    assert len(tiles) == 9
    assert np.array_equal(merge_tiles(tiles, metadata), array)


def test_merge_tiles_rejects_uncovered_output() -> None:
    with pytest.raises(ValueError, match="cover"):
        merge_tiles(
            [np.ones((2, 2), dtype=np.uint8)],
            {"shape": (3, 3), "boxes": [(0, 0, 2, 2)]},
        )


def test_merge_tiles_rejects_dtype_and_channel_mismatch() -> None:
    metadata = {"shape": (2, 3, 1), "boxes": [(0, 0, 2, 2), (1, 0, 3, 2)]}
    with pytest.raises(ValueError, match="dtype"):
        merge_tiles(
            [
                np.zeros((2, 2, 1), dtype=np.uint8),
                np.zeros((2, 2, 1), dtype=np.float32),
            ],
            metadata,
        )
    with pytest.raises(ValueError, match="channel"):
        merge_tiles(
            [
                np.zeros((2, 2, 1), dtype=np.uint8),
                np.zeros((2, 2, 3), dtype=np.uint8),
            ],
            metadata,
        )


@pytest.mark.parametrize(
    "tile",
    [
        np.full((2, 2), 1 + 2j, dtype=np.complex64),
        np.full((2, 2), 1, dtype=object),
        np.full((2, 2), "1", dtype=np.str_),
    ],
    ids=["complex", "object", "string"],
)
def test_merge_tiles_rejects_non_real_numeric_tiles(tile: np.ndarray) -> None:
    metadata = {"shape": (2, 2), "boxes": [(0, 0, 2, 2)]}
    with pytest.raises(ValueError, match="tile must have"):
        merge_tiles([tile], metadata)


@pytest.mark.parametrize(
    "grid,overlap",
    [
        (True, 0.0),
        (2, float("nan")),
        ((2, 0), 0.0),
        ((1, 2, 3), 0.0),
        (2, 1.0),
    ],
)
def test_split_tiles_rejects_invalid_grid_or_overlap(
    grid: object,
    overlap: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        split_tiles(
            np.zeros((4, 4), dtype=np.uint8),
            grid=grid,
            overlap=overlap,
        )
