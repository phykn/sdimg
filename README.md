# sdimg

Small, function-based image and mask processing library built on
`numpy.ndarray`. The public API is pure functions and requires Python 3.12+.

## Install

```bash
pip install sdimg
```

## Modules

- `sdimg.image`: image conversion, filtering, enhancement, normalization,
  deterministic array IDs, lossless string codecs, and file I/O.
- `sdimg.mask`: binary conversion, geometry, connected components,
  morphology, hulls, and distance transforms.
- `sdimg.spatial`: crop, pad, resize, rotate, flip, and tile split/merge.
- `sdimg.segment`: Otsu thresholding and GrabCut refinement.

Public imports are flat within each domain:

```python
from sdimg.image import apply_gaussian_blur, equalize_histogram
from sdimg.mask import apply_morphology
from sdimg.segment import refine_grabcut
```

The implementation keeps representation-specific responsibilities inside their
domains: image processing owns visual/alpha channel handling, spatial operations
own dtype restoration, and segmentation owns its bbox and ROI lifecycle. Import
public functions from the flat domain packages rather than implementation
modules.

## Core Contracts

- Inputs are `numpy.ndarray` values with real numeric or boolean dtype.
- Images use non-empty shape `(H, W)` or `(H, W, C)` with `C in {1, 2, 3, 4}`.
- Color images are RGB. Convert OpenCV BGR inputs before calling `sdimg`.
- Channel meanings are grayscale, grayscale+alpha, RGB, and RGBA.
- `sdimg.image` processing returns `np.uint8`. Filters and enhancement preserve
  channel shape and leave alpha content unchanged.
- Masks use non-empty shape `(H, W)` and values `bool`, `{0, 1}`, or `{0, 255}`.
  Mask processing returns binary `np.uint8` in `{0, 1}`; distance transforms
  return `np.float32`.
- `sdimg.spatial` operations preserve input dtype and channel shape.
- `refine_grabcut` accepts an image-sized binary mask and returns a binary mask
  with the same spatial shape. It derives and restores its ROI internally.
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
from sdimg.mask import apply_morphology, extract_roi
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
The lossless WebP string codec preserves RGBA alpha and ignores
grayscale+alpha's alpha channel.
