# sdimg

Small, function-based image and mask processing library built on
`numpy.ndarray`. The public API is pure functions and requires Python 3.12+.

## Install

```bash
pip install sdimg
```

## Architecture

| Path | Responsibility |
| --- | --- |
| `sdimg.core` | Shared array, shape, parameter, and bbox contracts. It contains no image-processing algorithms. |
| `sdimg.image` | Image representation conversion, filtering, enhancement, identity, codecs, and file I/O. Channel and Pillow adapters stay in this domain. |
| `sdimg.mask` | Binary-mask conversion, measurement, morphology, connected components, hulls, and distance transforms. |
| `sdimg.spatial` | Dtype-preserving crop, pad, resize, transform, and tiling operations. `tile.split` owns tile layout; `tile.merge` owns validated numerical reconstruction. |
| `sdimg.segment` | Algorithms that orchestrate image, mask, and spatial operations: Otsu thresholding and GrabCut refinement. |

The dependency direction is `core <- {image, mask, spatial} <- segment`.
Representation-specific helpers stay with the domain that owns the
representation; cross-domain workflows belong in `segment`.

Public imports are flat within each domain:

```python
from sdimg.image import apply_gaussian_blur, equalize_histogram
from sdimg.mask import apply_morphology
from sdimg.segment import refine_grabcut
```

Import public functions from the flat domain packages rather than implementation
modules.

## Core Contracts

- Processing inputs are `numpy.ndarray` values with real numeric or boolean dtype.
- Images use non-empty shape `(H, W)` or `(H, W, C)` with `C in {1, 2, 3, 4}`.
- Color images are RGB. Convert OpenCV BGR inputs before calling `sdimg`.
- Channel meanings are grayscale, grayscale+alpha, RGB, and RGBA.
- `sdimg.image` processing returns `np.uint8`. Filters and enhancement preserve
  channel shape and leave alpha content unchanged. Color denoising follows the
  RGB contract; setting both denoising strengths to zero is an exact converted
  no-op.
- Masks use non-empty shape `(H, W)` and values `bool`, `{0, 1}`, or `{0, 255}`.
  Mask processing returns binary `np.uint8` in `{0, 1}`; distance transforms
  return `np.float32`. Two-dimensional morphology kernel sizes use
  `(width, height)` order.
- `sdimg.spatial` operations preserve input dtype and channel shape. `resize`
  and tile merging reject integer values outside the safe range
  `[-2**53, 2**53]` instead of silently changing them.
- `refine_grabcut` accepts an image-sized binary mask and returns a binary mask
  with the same spatial shape. Its `margin` expands the initial bounding box
  over real source pixels, so refinement may add foreground outside that box.
- Points use `(x, y)`. Bounding boxes use
  `(xmin, ymin, xmax, ymax)`, minimum-inclusive and maximum-exclusive.
- Empty mask geometry returns `None` where no geometry exists.

## Error Policy

- `TypeError`: wrong Python input type.
- `ValueError`: invalid shape, dtype, parameter, mask value, or bbox.
- `RuntimeError`: wrapped OpenCV, Pillow, or third-party failure.

## Quick Example

```python
import numpy as np

from sdimg.image import apply_gaussian_blur, equalize_histogram
from sdimg.mask import apply_morphology
from sdimg.segment import refine_grabcut

image = np.random.default_rng(0).integers(
    0,
    256,
    (128, 128, 3),
    dtype=np.uint8,
)
mask = np.zeros((128, 128), dtype=np.uint8)
mask[32:96, 40:88] = 1

image = equalize_histogram(image)
image = apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=1.2)
mask = apply_morphology(mask, operation="open", kernel_size=(3, 3))

refined = refine_grabcut(image, mask)
```

## Supporting Image Utilities

```python
from sdimg.image import (
    decode_image,
    encode_image,
    make_array_id,
    read_image,
    write_image,
)

image = read_image("input.tif")
image_id = make_array_id(image, prefix="img_")

payload = encode_image(image)
restored = decode_image(payload)
write_image(f"{image_id}.png", restored)
```

`read_image` returns RGB `uint8`. High-bit-depth and floating images are scaled
explicitly before RGB conversion. `write_image` writes RGB and ignores alpha.
`make_array_id` accepts any shape and any non-object dtype, and includes the
full dtype definition, original shape, and content. Structured dtype field names
and layout therefore participate in the ID.
The lossless WebP string codec preserves RGBA alpha and ignores
grayscale+alpha's alpha channel.
