from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sdimg.image import imread, imwrite


def test_imread_imwrite_round_trips_rgb(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[0, 0] = [255, 0, 0]

    imwrite(path, image)
    out = imread(path)

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_imwrite_accepts_2d_grayscale_and_reads_back_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)

    imwrite(path, image)
    out = imread(path)

    assert out.shape == (10, 10, 3)
    assert np.array_equal(out, np.repeat(image[..., None], 3, axis=2))


def test_imwrite_accepts_single_channel_3d_and_reads_back_rgb(tmp_path: Path) -> None:
    path = tmp_path / "gray3d.png"
    image = np.full((10, 10, 1), 128, dtype=np.uint8)

    imwrite(path, image)
    out = imread(path)

    assert out.shape == (10, 10, 3)
    assert np.array_equal(out, np.repeat(image, 3, axis=2))


def test_imwrite_accepts_rgba_and_reads_back_rgb(tmp_path: Path) -> None:
    path = tmp_path / "rgba.png"
    image = np.zeros((10, 10, 4), dtype=np.uint8)
    image[..., 0] = 255
    image[..., 3] = 128

    imwrite(path, image)
    out = imread(path)

    assert out.shape == (10, 10, 3)
    assert np.array_equal(out, image[..., :3])


def test_imwrite_rejects_non_ndarray_input(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        imwrite(tmp_path / "bad.png", "not-an-array")  # type: ignore[arg-type]


def test_imwrite_rejects_non_uint8(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uint8"):
        imwrite(tmp_path / "bad.png", np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3, 4, 1), dtype=np.uint8),
        np.zeros((4, 4, 2), dtype=np.uint8),
        np.zeros((4, 4, 5), dtype=np.uint8),
    ],
)
def test_imwrite_rejects_unsupported_shapes(tmp_path: Path, image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        imwrite(tmp_path / "bad.png", image)


def test_imread_wraps_pillow_failures(tmp_path: Path) -> None:
    path = tmp_path / "not-image.png"
    path.write_text("not image", encoding="utf-8")

    with pytest.raises(RuntimeError, match="imread failed"):
        imread(path)


def test_imwrite_wraps_save_failures(tmp_path: Path) -> None:
    target_dir = tmp_path / "directory-target"
    target_dir.mkdir()

    with pytest.raises(RuntimeError, match="imwrite failed"):
        imwrite(target_dir, np.zeros((4, 4, 3), dtype=np.uint8))


def test_imread_scales_uint16_tiff_to_uint8_rgb(tmp_path: Path) -> None:
    path = tmp_path / "source.tif"
    data = np.array([[0, 255, 256, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(data).save(path)

    out = imread(path)

    assert out.dtype == np.uint8
    assert out.shape == (1, 5, 3)
    assert out[0, :, 0].tolist() == [0, 1, 1, 128, 255]
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 0], out[..., 2])


def test_imread_scales_float_tiff_to_uint8_rgb(tmp_path: Path) -> None:
    path = tmp_path / "float.tif"
    data = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    Image.fromarray(data).save(path)

    out = imread(path)

    assert out.dtype == np.uint8
    assert out.shape == (1, 3, 3)
    assert out[0, :, 0].tolist() == [0, 128, 255]
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 0], out[..., 2])
