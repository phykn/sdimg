# sdimg.image I/O, ID, and String Codec Design

## Goal

Add the useful parts of `phykn/imid`, `phykn/imstr`, and `phykn/imrw` into
`sdimg.image` as first-class image utilities.

The public API should stay small, function-based, and consistent with the
existing `sdimg` contracts:

- inputs are `numpy.ndarray` where array data is involved;
- images use shape `(H, W)` or `(H, W, C)` with `C in {1, 2, 3, 4}`;
- color image data is RGB;
- public image outputs are `np.uint8`.

## Public API

The new functions are exported from `sdimg.image`:

```python
from sdimg.image import decode, encode, get_id, imread, imwrite
```

### `get_id(arr, *, prefix="", length=8) -> str`

Create a deterministic MD5-based ID for any `numpy.ndarray`.

The digest includes:

- `arr.dtype.str`;
- `arr.shape`;
- contiguous byte content.

This preserves the `imid` behavior and intentionally does not restrict input
to valid image shapes. It remains useful for masks, intermediate arrays, and
non-image ndarray values.

Validation:

- raise `TypeError` when `arr` is not a `numpy.ndarray`;
- raise `TypeError` when `length` is not an `int` or is `bool`;
- raise `ValueError` when `length` is outside `1..32`.

### `encode(image, *, method=0, quality=0) -> str`

Serialize a `uint8` image array to a base64 string using Pillow WebP lossless
encoding.

Accepted shapes:

- `(H, W)`;
- `(H, W, 1)`;
- `(H, W, 3)`;
- `(H, W, 4)`.

`(H, W, 2)` is a valid `sdimg` image shape, but it represents grayscale plus
alpha. The codec will not support it because Pillow WebP does not preserve that
two-channel shape directly and the old `imstr` contract did not support it.

Shape restoration uses a short payload prefix before the WebP bytes. Grayscale
inputs, including `(H, W, 1)`, are stored as grayscale and decode to `(H, W)`.
RGB and RGBA inputs decode to their original 3-channel or 4-channel array shape.

Validation:

- raise `TypeError` when `image` is not a `numpy.ndarray`;
- raise `ValueError` when dtype is not `np.uint8`;
- raise `ValueError` for unsupported shape or channel count;
- raise `ValueError` when Pillow serialization fails.

### `decode(encoded) -> np.ndarray`

Deserialize a string produced by `encode`.

Output:

- `np.uint8`;
- `(H, W)` for grayscale payloads;
- `(H, W, 3)` or `(H, W, 4)` for color payloads.

Validation:

- raise `TypeError` when `encoded` is not `str`;
- raise `ValueError` for invalid base64, invalid prefix, or invalid WebP data.

### `imread(path) -> np.ndarray`

Read an image file with Pillow and return RGB `np.uint8` data with shape
`(H, W, 3)`.

This follows `imrw`: source modes such as grayscale, RGBA, palette, CMYK, and
integer or float TIFF are converted through Pillow's `RGB` conversion.

Validation and lower-level failures should use the existing project policy:

- path-like errors and Pillow open/convert errors are wrapped as `RuntimeError`
  with context from `imread`.

### `imwrite(path, image, **kwargs) -> None`

Write a `uint8` image array with Pillow and save as RGB.

Accepted shapes:

- `(H, W)`;
- `(H, W, 1)`;
- `(H, W, 3)`;
- `(H, W, 4)`.

`(H, W, 2)` is rejected for the same reason as `encode`: preserving
grayscale-plus-alpha as an RGB output would silently drop the documented alpha
semantics. Callers can explicitly convert with `to_rgb` or select channels
before writing.

`**kwargs` are forwarded to `PIL.Image.Image.save`.

Validation:

- raise `TypeError` when `image` is not a `numpy.ndarray`;
- raise `ValueError` when dtype is not `np.uint8`;
- raise `ValueError` for unsupported shape or channel count;
- raise `RuntimeError` when Pillow save fails.

## Internal Structure

Add small implementation files under `src/sdimg/image/`:

- `id.py` for `get_id`;
- `string.py` for `encode` and `decode`;
- `io.py` for `imread` and `imwrite`.

Update `src/sdimg/image/__init__.py` to export the new functions. Do not add
aliases such as `get_image_id` or `encode_image`; the original API names are
short and match the requested source projects.

Add `pillow` to runtime dependencies in `pyproject.toml`.

## Error Handling

Keep existing `sdimg` error categories:

- `TypeError`: wrong Python input type;
- `ValueError`: invalid dtype, shape, parameter, or encoded payload;
- `RuntimeError`: wrapped Pillow or lower-level I/O failures.

`encode` and `decode` use `ValueError` for invalid payloads because the caller
can correct the provided value. `imread` and `imwrite` use `RuntimeError` for
Pillow failures because they are lower-level file/codec failures.

## Tests

Add focused tests under `tests/image/`.

Required coverage:

- `get_id` is deterministic and changes with dtype, shape, and content;
- `get_id` handles non-contiguous arrays like contiguous copies;
- `get_id` validates `arr` and `length`;
- `encode`/`decode` round-trip 2D grayscale, 3D RGB, and 3D RGBA `uint8`;
- `encode` accepts `(H, W, 1)` and decodes it as 2D grayscale;
- `encode` rejects non-`uint8`, invalid ndim, channel count 2, and invalid
  channel count;
- `decode` rejects invalid base64, invalid prefix, and invalid WebP payload;
- `imread`/`imwrite` round-trip RGB through a temporary file;
- `imwrite` accepts 2D grayscale, `(H, W, 1)`, RGB, and RGBA and reads back RGB;
- `imwrite` rejects non-`uint8`, invalid ndim, channel count 2, and invalid
  channel count;
- `imread` handles source files according to Pillow RGB conversion.

Verification command:

```bash
PYTHONPATH=src pytest -q
```

## Documentation

Update `README.md` module listing and core contracts only where needed:

- add `get_id`, `encode`, `decode`, `imread`, and `imwrite` to `sdimg.image`;
- note that Pillow-backed file I/O reads RGB `uint8` arrays and writes RGB
  images from accepted `uint8` image arrays.
