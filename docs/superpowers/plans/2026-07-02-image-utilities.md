# Image Utilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `get_id`, `encode`, `decode`, `imread`, and `imwrite` to `sdimg.image`.

**Architecture:** Keep the public API flat in `sdimg.image`, but split implementations into small files by responsibility: ID generation, WebP/base64 string codec, and Pillow file I/O. Use the existing `core` validators for ndarray type checks and keep `sdimg` error categories.

**Tech Stack:** Python 3.13 in `.venv`, NumPy, Pillow, pytest, existing OpenCV dependency.

---

## File Structure

- Create `src/sdimg/image/id.py`: deterministic ndarray ID generation.
- Create `src/sdimg/image/string.py`: WebP lossless + base64 image encode/decode.
- Create `src/sdimg/image/io.py`: Pillow-backed `imread` and `imwrite`.
- Modify `src/sdimg/image/__init__.py`: export new public functions.
- Modify `pyproject.toml`: add `pillow` runtime dependency.
- Modify `README.md`: document the new `sdimg.image` functions.
- Create `tests/image/test_image_id_contracts.py`: ID behavior and validation.
- Create `tests/image/test_image_string_contracts.py`: string codec behavior and validation.
- Create `tests/image/test_image_io_contracts.py`: image read/write behavior and validation.

## Baseline

- [ ] **Step 1: Verify the starting test suite**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest -q
```

Expected: `147 passed`.

### Task 1: Add `get_id`

**Files:**
- Create: `src/sdimg/image/id.py`
- Modify: `src/sdimg/image/__init__.py`
- Test: `tests/image/test_image_id_contracts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/image/test_image_id_contracts.py`:

```python
import numpy as np
import pytest

from sdimg.image import get_id


def test_get_id_is_deterministic() -> None:
    arr = np.zeros((4, 4), dtype=np.uint8)

    assert get_id(arr) == get_id(arr)


def test_get_id_changes_with_content_shape_and_dtype() -> None:
    base = np.zeros((4, 4), dtype=np.uint8)

    assert get_id(base) != get_id(np.ones((4, 4), dtype=np.uint8))
    assert get_id(base) != get_id(np.zeros((2, 8), dtype=np.uint8))
    assert get_id(base) != get_id(np.zeros((4, 4), dtype=np.float32))


def test_get_id_handles_prefix_length_and_non_contiguous_arrays() -> None:
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    sliced = arr[::2]

    assert get_id(arr, prefix="img_").startswith("img_")
    assert len(get_id(arr, length=16)) == 16
    assert get_id(sliced) == get_id(np.ascontiguousarray(sliced))


def test_get_id_rejects_non_ndarray_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        get_id([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize("length", [0, -1, 33])
def test_get_id_rejects_invalid_length_value(length: int) -> None:
    with pytest.raises(ValueError, match="length must be between 1 and 32"):
        get_id(np.zeros((2, 2), dtype=np.uint8), length=length)


@pytest.mark.parametrize("length", [True, 1.0, "8"])
def test_get_id_rejects_invalid_length_type(length: object) -> None:
    with pytest.raises(TypeError, match="length must be an int"):
        get_id(np.zeros((2, 2), dtype=np.uint8), length=length)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/image/test_image_id_contracts.py -q
```

Expected: collection fails with `ImportError` because `sdimg.image` does not export `get_id`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/sdimg/image/id.py`:

```python
import hashlib

import numpy as np

from ..core.validate import ensure_ndarray


def get_id(arr: np.ndarray, *, prefix: str = "", length: int = 8) -> str:
    arr = ensure_ndarray(arr, name="arr")
    if not isinstance(length, int) or isinstance(length, bool):
        raise TypeError("length must be an int.")
    if not 1 <= length <= 32:
        raise ValueError("length must be between 1 and 32.")

    contiguous_arr = np.ascontiguousarray(arr)
    hasher = hashlib.md5()
    hasher.update(contiguous_arr.dtype.str.encode("ascii"))
    hasher.update(np.asarray(contiguous_arr.shape, dtype=np.int64).tobytes())
    hasher.update(memoryview(contiguous_arr))
    return prefix + hasher.hexdigest()[:length]
```

Modify `src/sdimg/image/__init__.py`:

```python
from .brightness_contrast import adjust_brightness_contrast
from .blur import gaussian_blur, median_blur
from .convert import is_image, to_gray, to_rgb, to_uint8
from .denoise import denoise
from .id import get_id
from .norm import clahe_norm, hist_norm, minmax_norm, zscore_norm
from .sharpen import sharpen
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/image/test_image_id_contracts.py -q
```

Expected: all tests in `test_image_id_contracts.py` pass.

### Task 2: Add `encode` and `decode`

**Files:**
- Create: `src/sdimg/image/string.py`
- Modify: `src/sdimg/image/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/image/test_image_string_contracts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/image/test_image_string_contracts.py`:

```python
import base64

import numpy as np
import pytest

from sdimg.image import decode, encode


def test_encode_decode_round_trips_2d_grayscale() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)

    out = decode(encode(image))

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_encode_decode_round_trips_rgb() -> None:
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)

    out = decode(encode(image))

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_encode_decode_round_trips_rgba() -> None:
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 0] = 255
    image[..., 3] = np.arange(64, dtype=np.uint8).reshape(8, 8)

    out = decode(encode(image))

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_encode_accepts_single_channel_3d_and_decodes_to_2d() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5, 1)

    out = decode(encode(image))

    assert out.shape == (5, 5)
    assert np.array_equal(out, image[..., 0])


def test_encode_rejects_non_ndarray_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        encode("not-an-array")  # type: ignore[arg-type]


def test_encode_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="uint8"):
        encode(np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3, 4, 1), dtype=np.uint8),
        np.zeros((4, 4, 2), dtype=np.uint8),
        np.zeros((4, 4, 5), dtype=np.uint8),
    ],
)
def test_encode_rejects_unsupported_shapes(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        encode(image)


def test_decode_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="str"):
        decode(b"not-a-string")  # type: ignore[arg-type]


def test_decode_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError, match="failed to deserialize array"):
        decode("not-base64")


def test_decode_rejects_invalid_prefix() -> None:
    encoded = base64.b64encode(b"Xabc").decode("utf-8")

    with pytest.raises(ValueError, match="invalid payload prefix"):
        decode(encoded)


def test_decode_rejects_invalid_webp_payload() -> None:
    encoded = base64.b64encode(b"Rnot-webp").decode("utf-8")

    with pytest.raises(ValueError, match="failed to deserialize array"):
        decode(encoded)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/image/test_image_string_contracts.py -q
```

Expected: collection fails with `ImportError` because `sdimg.image` does not export `encode` or `decode`.

- [ ] **Step 3: Add the dependency**

Modify `pyproject.toml` dependencies:

```toml
dependencies = [
    "numpy",
    "opencv-python-headless",
    "concave-hull",
    "pillow",
]
```

Install the package in the `.venv` if Pillow is missing:

```bash
.\.venv\Scripts\python.exe -m pip install -e .
```

Expected: install exits with status 0.

- [ ] **Step 4: Write the minimal implementation**

Create `src/sdimg/image/string.py`:

```python
import base64
import io

import numpy as np
from PIL import Image

from ..core.validate import ensure_ndarray


def encode(image: np.ndarray, *, method: int = 0, quality: int = 0) -> str:
    image = ensure_ndarray(image, name="image")
    payload_prefix, image_for_pillow = _prepare_encode_image(image)

    try:
        buffer = io.BytesIO()
        Image.fromarray(image_for_pillow).save(
            buffer,
            format="WebP",
            lossless=True,
            method=method,
            quality=quality,
        )
    except Exception as exc:
        raise ValueError(f"failed to serialize array: {exc}") from exc

    return base64.b64encode(payload_prefix + buffer.getvalue()).decode("utf-8")


def decode(encoded: str) -> np.ndarray:
    if not isinstance(encoded, str):
        raise TypeError("encoded must be a str.")

    try:
        payload = base64.b64decode(encoded, validate=True)
        prefix = payload[:1]
        webp_data = payload[1:]
        if prefix not in {b"L", b"R", b"A", b"C"}:
            raise ValueError("invalid payload prefix")

        with Image.open(io.BytesIO(webp_data)) as image:
            decoded = np.array(image)

        if prefix == b"L" and decoded.ndim == 3:
            return decoded[..., 0]
        if prefix == b"R" and decoded.ndim == 3 and decoded.shape[2] >= 3:
            return decoded[..., :3]
        return decoded
    except Exception as exc:
        raise ValueError(f"failed to deserialize array: {exc}") from exc


def _prepare_encode_image(image: np.ndarray) -> tuple[bytes, np.ndarray]:
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if image.ndim == 2:
        return b"L", image
    if image.ndim != 3:
        raise ValueError("image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).")

    channels = image.shape[2]
    if channels == 1:
        return b"L", image[..., 0]
    if channels == 3:
        return b"R", image
    if channels == 4:
        return b"A", image
    raise ValueError("image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).")
```

Modify `src/sdimg/image/__init__.py`:

```python
from .brightness_contrast import adjust_brightness_contrast
from .blur import gaussian_blur, median_blur
from .convert import is_image, to_gray, to_rgb, to_uint8
from .denoise import denoise
from .id import get_id
from .norm import clahe_norm, hist_norm, minmax_norm, zscore_norm
from .sharpen import sharpen
from .string import decode, encode
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/image/test_image_string_contracts.py -q
```

Expected: all tests in `test_image_string_contracts.py` pass.

### Task 3: Add `imread` and `imwrite`

**Files:**
- Create: `src/sdimg/image/io.py`
- Modify: `src/sdimg/image/__init__.py`
- Test: `tests/image/test_image_io_contracts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/image/test_image_io_contracts.py`:

```python
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


def test_imread_matches_pillow_rgb_conversion_for_uint16_tiff(tmp_path: Path) -> None:
    path = tmp_path / "source.tif"
    data = np.linspace(0, 65535, 25, dtype=np.uint16).reshape(5, 5)
    Image.fromarray(data).save(path)

    with Image.open(path) as image:
        expected = np.array(image.convert("RGB"))

    out = imread(path)

    assert out.dtype == np.uint8
    assert np.array_equal(out, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/image/test_image_io_contracts.py -q
```

Expected: collection fails with `ImportError` because `sdimg.image` does not export `imread` or `imwrite`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/sdimg/image/io.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.validate import ensure_ndarray


def imread(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"imread failed: {exc}") from exc


def imwrite(path: str | Path, image: np.ndarray, **kwargs: object) -> None:
    image = ensure_ndarray(image, name="image")
    image_for_pillow = _prepare_write_image(image)

    try:
        Image.fromarray(image_for_pillow).convert("RGB").save(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"imwrite failed: {exc}") from exc


def _prepare_write_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).")

    channels = image.shape[2]
    if channels == 1:
        return image[..., 0]
    if channels in {3, 4}:
        return image
    raise ValueError("image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).")
```

Modify `src/sdimg/image/__init__.py`:

```python
from .brightness_contrast import adjust_brightness_contrast
from .blur import gaussian_blur, median_blur
from .convert import is_image, to_gray, to_rgb, to_uint8
from .denoise import denoise
from .id import get_id
from .io import imread, imwrite
from .norm import clahe_norm, hist_norm, minmax_norm, zscore_norm
from .sharpen import sharpen
from .string import decode, encode
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/image/test_image_io_contracts.py -q
```

Expected: all tests in `test_image_io_contracts.py` pass.

### Task 4: Update README and full verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README module list**

Modify the `sdimg.image` bullet to include the new functions:

```markdown
- `sdimg.image`: `hist_norm`, `clahe_norm`, `minmax_norm`, `zscore_norm`, `gaussian_blur`, `median_blur`, `denoise`, `sharpen`, `adjust_brightness_contrast`, `to_gray`, `to_rgb`, `to_uint8`, `get_id`, `encode`, `decode`, `imread`, `imwrite`, `is_image`.
```

- [ ] **Step 2: Update README contracts**

Add this bullet under Core Contracts:

```markdown
- Pillow-backed file I/O reads images as RGB `np.uint8` arrays with shape `(H, W, 3)` and writes accepted `uint8` image arrays as RGB files.
```

- [ ] **Step 3: Run focused image tests**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/image -q
```

Expected: all image tests pass.

- [ ] **Step 4: Run the full suite**

Run:

```bash
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Inspect the diff**

Run:

```bash
git diff -- src/sdimg/image pyproject.toml README.md tests/image
```

Expected: diff only includes the requested image utility code, tests, dependency, and README updates.
