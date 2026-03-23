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


def test_split_returns_only_patches_when_return_meta_is_false() -> None:
    src = np.arange(8 * 10, dtype=np.uint8).reshape(8, 10)
    patches = split(src, n=2, overlap=0.25, return_meta=False)
    
    # Should be a list of ndarrays, not a tuple
    assert isinstance(patches, list)
    assert all(isinstance(p, np.ndarray) for p in patches)
    assert len(patches) > 0
