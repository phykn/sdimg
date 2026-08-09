import importlib

import numpy as np
import pytest

from sdimg.spatial import merge_tiles, split_tiles


def test_tile_package_preserves_flat_and_package_imports() -> None:
    tile_module = importlib.import_module("sdimg.spatial.tile")

    assert tile_module.merge_tiles is merge_tiles
    assert tile_module.split_tiles is split_tiles


@pytest.mark.parametrize(
    "array",
    [
        np.arange(24, dtype=np.uint8).reshape(4, 6),
        np.arange(24).reshape(4, 6) % 2 == 0,
        np.array(
            [
                np.nan,
                np.inf,
                -np.inf,
                1.5,
                2.5,
                3.5,
                4.5,
                5.5,
                6.5,
                7.5,
                8.5,
                9.5,
                10.5,
                11.5,
                12.5,
                13.5,
            ],
            dtype=np.float32,
        ).reshape(4, 4),
        np.arange(72, dtype=np.uint8).reshape(4, 6, 3),
    ],
    ids=["uint8", "bool", "float-nonfinite", "channels"],
)
def test_merge_tiles_direct_stitches_nonoverlapping_tiles(
    monkeypatch: pytest.MonkeyPatch,
    array: np.ndarray,
) -> None:
    merge_module = importlib.import_module("sdimg.spatial.tile.merge")

    def fail_if_aggregated(*args: object, **kwargs: object) -> None:
        raise AssertionError("non-overlapping tiles must bypass numeric aggregation")

    monkeypatch.setattr(merge_module, "_merge_integer_tiles", fail_if_aggregated)
    monkeypatch.setattr(merge_module, "_merge_real_tiles", fail_if_aggregated)
    tiles, metadata = split_tiles(
        array,
        grid=(2, 2),
        return_metadata=True,
    )

    out = merge_tiles(tiles, metadata)

    assert out.dtype == array.dtype
    assert out.flags.c_contiguous
    if np.issubdtype(array.dtype, np.floating):
        assert np.array_equal(out, array, equal_nan=True)
    else:
        assert np.array_equal(out, array)
    assert not any(np.shares_memory(out, tile) for tile in tiles)
