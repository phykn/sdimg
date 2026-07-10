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
