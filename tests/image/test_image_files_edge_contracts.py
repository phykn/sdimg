from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sdimg.image import read_image, write_image


def test_read_scales_uint16_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "source.tif"
    source = np.linspace(0, 65535, 25, dtype=np.uint16).reshape(5, 5)
    Image.fromarray(source).save(path)
    out = read_image(path)
    assert out.dtype == np.uint8
    assert out.shape == (5, 5, 3)
    assert out.min() == 0 and out.max() == 255


def test_write_read_roundtrip_is_rgb_and_ignores_alpha(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    image = np.zeros((4, 5, 4), dtype=np.uint8)
    image[..., :3] = [10, 20, 30]
    image[..., 3] = 77
    write_image(path, image)
    assert np.array_equal(read_image(path), image[..., :3])


def test_read_scales_unit_float_to_rgb(tmp_path: Path) -> None:
    path = tmp_path / "source.tif"
    source = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)
    Image.fromarray(source, mode="F").save(path)
    out = read_image(path)
    assert out.min() == 0 and out.max() == 255


def test_read_and_write_wrap_pillow_failures(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not image", encoding="utf-8")
    with pytest.raises(RuntimeError, match="read_image failed"):
        read_image(invalid)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="write_image failed"):
        write_image(directory, np.zeros((2, 2, 3), dtype=np.uint8))
