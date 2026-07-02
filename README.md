# sdimg

Small, function-based image and mask processing library built on `numpy.ndarray`.
The public API is pure functions and requires Python 3.12+.

## Install

```bash
pip install sdimg
```

## Modules

- `sdimg.image`: `hist_norm`, `clahe_norm`, `minmax_norm`, `zscore_norm`, `gaussian_blur`, `median_blur`, `denoise`, `sharpen`, `adjust_brightness_contrast`, `to_gray`, `to_rgb`, `to_uint8`, `get_id`, `encode`, `decode`, `imread`, `imwrite`, `is_image`.
- `sdimg.mask`: `morphology`; `convex_hull`, `concave_hull`; `extract_edge`, `distance_transform`, `pick_largest`, `fill_holes`; `get_box_from_mask`, `get_box_from_coords`, `get_coords`, `get_centroid`, `get_roi_size`, `get_box_size`, `to_roi_box`; `to_mask`, and `is_mask`.
- `sdimg.spatial`: `resize`, `resize_keep_ratio`, `crop`, `pad_to_square`, `rotate`, `flip`, `split`, and `merge`.
- `sdimg.fusion`: `otsu_threshold` and `grabcut`.

## Core Contracts

- Inputs must be `numpy.ndarray`.
- Images use shape `(H, W)` or `(H, W, C)` with `C in 1..4`.
- Color images are RGB. If you read with `cv2.imread`, convert BGR to RGB first with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- Pillow-backed file I/O reads images as RGB `np.uint8` arrays with shape `(H, W, 3)` and writes accepted `uint8` image arrays as RGB files.
- Channel counts mean: `1` grayscale, `2` grayscale + alpha, `3` RGB, and `4` RGBA. Alpha channels are ignored by `to_gray` and `to_rgb`.
- Masks use shape `(H, W)` and binary values: `bool`, `{0, 1}`, or `{0, 255}`.
- Output images are `np.uint8`; output masks are binary `np.uint8` in `{0, 1}`.
- BBox format is `(wmin, hmin, wmax, hmax)`, width-first, min-inclusive, max-exclusive.
- Empty masks return `None` from `to_roi_box`, `get_box_from_mask`, `get_box_from_coords`, and `get_centroid`.

## Error Policy

- `TypeError`: wrong input type.
- `ValueError`: invalid shape, params, mask values, or bbox.
- `RuntimeError`: wrapped lower-level failures from `cv2` or internal processing.

## Quick Example

```python
import numpy as np

from sdimg.fusion import grabcut
from sdimg.image import gaussian_blur, hist_norm
from sdimg.mask import morphology, to_roi_box

image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
mask = np.zeros((128, 128), dtype=np.uint8)
mask[32:96, 40:88] = 1

image = gaussian_blur(hist_norm(image), (5, 5), 1.2)
mask = morphology(mask, "open", (3, 3), 1)

roi_box = to_roi_box(mask)
if roi_box is not None:
    refined = grabcut(image=image, roi=roi_box["roi"], box=roi_box["box"])
```
