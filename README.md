# sdimg

Small, function-based image and mask processing library built on `numpy.ndarray`.

## Install

```bash
pip install sdimg
```

## Modules

- `sdimg.image`: normalize, blur, denoise, sharpen, color helpers
- `sdimg.mask`: binary mask cleanup, hull/edge/distance, bbox/ROI helpers
- `sdimg.spatial`: resize, crop, rotate/flip, pad, split/merge patches
- `sdimg.fusion`: `otsu_threshold` (OpenCV Otsu), `grabcut`

## Core Contracts

- Input arrays must be `numpy.ndarray`
- Images: shape `(H, W)` or `(H, W, C)` with `C in 1..4`
- **Color channel order: RGB.** Color images passed to `sdimg` must be in RGB order. `cv2.imread` returns BGR — callers using OpenCV I/O must convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before calling `sdimg` functions.
- **Channel-count semantics:**
  - `C == 1`: grayscale
  - `C == 2`: grayscale + alpha (alpha is ignored by `to_gray`/`to_rgb`)
  - `C == 3`: RGB
  - `C == 4`: RGBA (alpha is ignored by `to_gray`/`to_rgb`)
- Masks: shape `(H, W)`, binary values (`bool`, `{0,1}`, `{0,255}`)
- Output images are `np.uint8`
- Output masks are binary `np.uint8` in `{0, 1}`
- BBox format: `(wmin, hmin, wmax, hmax)`
- Empty-mask returns `None` for:
  - `to_roi_box`
  - `get_box_from_mask`
  - `get_box_from_coords`
  - `get_centroid`

## Error Policy

- `TypeError`: wrong input type (non-`numpy.ndarray`)
- `ValueError`: invalid shape, invalid params, invalid mask values, invalid bbox
- `RuntimeError`: wrapped lower-level failures (`cv2`, internal processing)

## Internal Structure

- `sdimg/_core/validate.py`: shared validators (`ensure_src`, `ensure_image`, `ensure_mask`, `ensure_bbox`)
- `sdimg/_core/types.py`: shared type aliases
- `sdimg/_core/errors.py`: shared error helpers

## Quick Example

```python
import numpy as np
from sdimg.image import hist_norm, gaussian_blur
from sdimg.mask import morphology, to_roi_box
from sdimg.fusion import grabcut

image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
mask = np.zeros((128, 128), dtype=np.uint8)
mask[32:96, 40:88] = 1

image = hist_norm(image)
image = gaussian_blur(image, (5, 5), 1.2)
mask = morphology(mask, "open", (3, 3), 1)

roi_box = to_roi_box(mask)
if roi_box is not None:
    refined = grabcut(image=image, roi=roi_box["roi"], box=roi_box["box"])
```

## Local Test

```bash
PYTHONPATH=. pytest -q
```
