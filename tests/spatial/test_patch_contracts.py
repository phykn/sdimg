import numpy as np
import pytest

from sdimg.spatial import merge, split


def test_split_merge_roundtrip_shape_dtype() -> None:
    src = np.arange(8 * 10, dtype=np.uint8).reshape(8, 10)
    patches, meta = split(src, n=2, overlap=0.25, return_meta=True)
    merged = merge(patches, meta)

    assert merged.shape == src.shape
    assert merged.dtype == np.uint8


def test_split_high_overlap_covers_full_extent() -> None:
    src = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    patches, meta = split(src, n=4, overlap=0.5, return_meta=True)

    assert len(patches) == 16
    boxes = meta["boxes"]
    assert boxes[0][0] == 0 and boxes[0][1] == 0
    assert boxes[-1][2] == 100 and boxes[-1][3] == 100

    for i in range(3):
        a, b = boxes[i], boxes[i + 1]
        patch_w = a[2] - a[0]
        step = b[0] - a[0]
        assert 1.0 - step / patch_w + 1e-9 >= 0.5


def test_split_non_divisible_length_with_overlap() -> None:
    src = np.arange(17 * 23, dtype=np.uint8).reshape(17, 23)
    patches, meta = split(src, n=3, overlap=0.3, return_meta=True)
    merged = merge(patches, meta)

    assert merged.shape == src.shape
    assert len(patches) == 9


def test_split_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        split(src, n=2)


def test_merge_rejects_invalid_meta_shape_type() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    patches, meta = split(src, n=2, return_meta=True)
    bad_meta = dict(meta)
    bad_meta["shape"] = list(meta["shape"])

    with pytest.raises(ValueError, match="meta\\['shape'\\] must be a tuple"):
        merge(patches, bad_meta)


def test_merge_rejects_box_out_of_bounds() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    patches, meta = split(src, n=2, return_meta=True)
    bad_meta = dict(meta)
    bad_meta["boxes"] = [(0, 0, 100, 4)] + list(meta["boxes"])[1:]

    with pytest.raises(ValueError, match="within output shape bounds"):
        merge(patches, bad_meta)


def test_merge_rejects_non_int_box_values() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    patches, meta = split(src, n=2, return_meta=True)
    bad_meta = dict(meta)
    bad_boxes = list(meta["boxes"])
    bad_boxes[0] = (0.0, 0, 4, 4)  # type: ignore[assignment]
    bad_meta["boxes"] = bad_boxes

    with pytest.raises(ValueError, match="tuple of 4 integers"):
        merge(patches, bad_meta)


def test_split_returns_only_patches_when_return_meta_is_false() -> None:
    src = np.arange(8 * 10, dtype=np.uint8).reshape(8, 10)
    patches = split(src, n=2, overlap=0.25, return_meta=False)

    assert isinstance(patches, list)
    assert all(isinstance(p, np.ndarray) for p in patches)
    assert len(patches) > 0


def test_split_rejects_invalid_n() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="n must be greater than 0"):
        split(src, n=0)


def test_split_rejects_invalid_overlap() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap must satisfy"):
        split(src, n=2, overlap=1.0)


def test_split_asymmetric_produces_correct_count() -> None:
    src = np.zeros((10, 20), dtype=np.uint8)
    patches, meta = split(src, n=(2, 1), return_meta=True)

    assert len(patches) == 2
    assert all(p.shape == (10, 10) for p in patches)


def test_split_asymmetric_roundtrip() -> None:
    src = np.arange(20 * 30, dtype=np.uint8).reshape(20, 30)
    patches, meta = split(src, n=(3, 2), return_meta=True)

    assert len(patches) == 6
    merged = merge(patches, meta)
    assert merged.shape == src.shape
    assert merged.dtype == np.uint8


def test_split_asymmetric_with_overlap() -> None:
    src = np.zeros((10, 100), dtype=np.uint8)
    patches, meta = split(src, n=(4, 1), overlap=0.25, return_meta=True)

    assert len(patches) == 4

    boxes = meta["boxes"]
    for i in range(3):
        a, b = boxes[i], boxes[i + 1]
        patch_w = a[2] - a[0]
        step = b[0] - a[0]
        assert 1.0 - step / patch_w + 1e-9 >= 0.25


def test_split_per_axis_overlap() -> None:
    src = np.zeros((40, 60), dtype=np.uint8)
    patches, meta = split(src, n=(3, 2), overlap=(0.5, 0.0), return_meta=True)

    assert len(patches) == 6
    merged = merge(patches, meta)
    assert merged.shape == src.shape


def test_split_tuple_n_rejects_zero_component() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="n must be greater than 0"):
        split(src, n=(0, 2))


def test_split_tuple_n_rejects_wrong_type() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(TypeError, match="n must be an int or a tuple of two ints"):
        split(src, n=(1.5, 2))  # type: ignore[arg-type]


def test_split_tuple_n_rejects_wrong_length() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(TypeError, match="n must be an int or a tuple of two ints"):
        split(src, n=(1, 2, 3))  # type: ignore[arg-type]


def test_split_tuple_overlap_rejects_out_of_range() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap must satisfy"):
        split(src, n=2, overlap=(0.5, 1.0))


def test_split_tuple_overlap_rejects_wrong_type() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(
        TypeError,
        match="overlap must be a float or a tuple of two floats",
    ):
        split(src, n=2, overlap="bad")  # type: ignore[arg-type]


def test_merge_rejects_empty_patches() -> None:
    with pytest.raises(ValueError, match="patches must not be empty"):
        merge([], {"shape": (8, 8), "boxes": []})
