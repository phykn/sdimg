import numpy as np
import pytest

from sdimg.spatial import (
    crop,
    flip,
    merge,
    pad_to_square,
    resize,
    resize_keep_ratio,
    rotate,
    split,
)


def test_resize_with_only_width_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize(src, width=40)

    assert out.shape[:2] == (20, 40)


def test_resize_requires_height_or_width() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="height or width must be provided"):
        resize(src)


def test_crop_rejects_invalid_bbox() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="bbox is out of bounds or invalid"):
        crop(src, bbox=(0, 0, 6, 4))


def test_pad_to_square_returns_box_and_square_shape() -> None:
    src = np.ones((3, 5), dtype=np.uint8)
    padded, box = pad_to_square(src, return_box=True)

    assert padded.shape == (5, 5)
    assert box == (0, 0, 5, 3)


def test_rotate_and_flip_contract() -> None:
    src = np.arange(6, dtype=np.uint8).reshape(2, 3)
    rotated = rotate(src, rotation=90)
    flipped = flip(src, direction="horizontal")

    assert rotated.shape == (3, 2)
    assert flipped.shape == src.shape


def test_rotate_rejects_invalid_angle() -> None:
    src = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="rotation must be one of"):
        rotate(src, rotation=45)


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
    # first box anchored at 0, last box ends exactly at length on both axes
    assert boxes[0][0] == 0 and boxes[0][1] == 0
    assert boxes[-1][2] == 100 and boxes[-1][3] == 100

    # adjacent patch overlap satisfies the requested ratio along the width axis
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


def test_pad_to_square_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        pad_to_square(src)


def test_crop_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        crop(src, bbox=(0, 0, 1, 1))


def test_resize_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        resize(src, width=10)


def test_resize_keep_ratio_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        resize_keep_ratio(src, long_side=10)


def test_rotate_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        rotate(src, rotation=90)


def test_flip_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        flip(src, direction="horizontal")


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


def test_crop_returns_independent_copy() -> None:
    src = np.arange(25, dtype=np.uint8).reshape(5, 5)
    cropped = crop(src, bbox=(1, 1, 3, 3))
    cropped[0, 0] = 255

    assert src[1, 1] != 255


def test_split_returns_only_patches_when_return_meta_is_false() -> None:
    src = np.arange(8 * 10, dtype=np.uint8).reshape(8, 10)
    patches = split(src, n=2, overlap=0.25, return_meta=False)

    # Should be a list of ndarrays, not a tuple
    assert isinstance(patches, list)
    assert all(isinstance(p, np.ndarray) for p in patches)
    assert len(patches) > 0


# --- resize_keep_ratio ---


def test_resize_keep_ratio_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize_keep_ratio(src, long_side=40)
    assert out.shape == (20, 40, 3)


def test_resize_keep_ratio_scales_down() -> None:
    src = np.zeros((100, 200), dtype=np.uint8)
    out = resize_keep_ratio(src, long_side=100)
    assert out.shape == (50, 100)


def test_resize_keep_ratio_rejects_non_positive_long_side() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="long_side must be greater than 0"):
        resize_keep_ratio(src, long_side=0)


# --- crop happy path ---


def test_crop_returns_correct_content() -> None:
    src = np.arange(25, dtype=np.uint8).reshape(5, 5)
    out = crop(src, bbox=(1, 2, 4, 4))
    assert out.shape == (2, 3)
    assert out[0, 0] == src[2, 1]
    assert out[1, 2] == src[3, 3]


# --- resize happy path ---


def test_resize_with_both_height_and_width() -> None:
    src = np.zeros((10, 20), dtype=np.uint8)
    out = resize(src, height=5, width=10)
    assert out.shape == (5, 10)


def test_resize_with_only_height_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize(src, height=20)
    assert out.shape[:2] == (20, 40)


# --- rotate happy path ---


def test_rotate_0_returns_same() -> None:
    src = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = rotate(src, rotation=0)
    assert np.array_equal(out, src)


def test_rotate_180_reverses_content() -> None:
    src = np.arange(4, dtype=np.uint8).reshape(2, 2)
    out = rotate(src, rotation=180)
    assert out[0, 0] == src[1, 1]
    assert out[1, 1] == src[0, 0]


def test_rotate_270_shape() -> None:
    src = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = rotate(src, rotation=270)
    assert out.shape == (3, 2)


# --- flip happy path ---


def test_flip_vertical_reverses_rows() -> None:
    src = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = flip(src, direction="vertical")
    assert out[0, 0] == 3
    assert out[1, 0] == 1


def test_flip_transpose_swaps_axes() -> None:
    src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    out = flip(src, direction="transpose")
    assert out.shape == (3, 2)
    assert out[0, 1] == 4
    assert out[2, 0] == 3


def test_flip_rejects_invalid_direction() -> None:
    src = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="direction must be one of"):
        flip(src, direction="diagonal")


# --- pad_to_square happy path ---


def test_pad_to_square_already_square() -> None:
    src = np.ones((5, 5), dtype=np.uint8)
    out = pad_to_square(src)
    assert out.shape == (5, 5)
    assert np.array_equal(out, src)


def test_pad_to_square_tall_image() -> None:
    src = np.ones((5, 3), dtype=np.uint8)
    out = pad_to_square(src)
    assert out.shape == (5, 5)


# --- split validation ---


def test_split_rejects_invalid_n() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="n must be greater than 0"):
        split(src, n=0)


def test_split_rejects_invalid_overlap() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap must satisfy"):
        split(src, n=2, overlap=1.0)


# --- merge validation ---


def test_merge_rejects_empty_patches() -> None:
    with pytest.raises(ValueError, match="patches must not be empty"):
        merge([], {"shape": (8, 8), "boxes": []})
