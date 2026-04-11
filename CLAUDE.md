# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`sdimg` — a small, function-based image and mask processing library built on `numpy.ndarray`. No classes; all public API is pure functions. Requires Python ≥ 3.12.

## Commands

```bash
# Install
pip install .

# Run all tests
PYTHONPATH=src pytest -q

# Run a single test
PYTHONPATH=src pytest tests/image/test_image_contracts.py::test_to_uint8_clips_and_rounds -v
```

## Architecture

Source lives in `src/sdimg/` with four public modules and one internal module:

- **`_core/`** — shared validators (`ensure_src`, `ensure_image`, `ensure_mask`, `ensure_bbox`), type aliases (`BBox`), and error factory helpers. Used by all other modules.
- **`image/`** — normalization, blur, denoise, sharpen, color/dtype conversion (`to_gray`, `to_rgb`, `to_uint8`, `is_image`).
- **`mask/`** — binary mask operations: morphology, hull, edge, distance, connected components, hole filling, and bbox/ROI helpers (`to_roi_box`, `get_box_from_mask`, `get_centroid`, `is_mask`, `to_mask`).
- **`spatial/`** — resize, crop, rotate/flip, pad, split/merge patches with overlap.
- **`fusion/`** — algorithms combining image+mask: `grabcut`, `otsu_threshold`.

Dependency flow: `_core` ← `image` ← `mask` ← `spatial` ← `fusion` (each layer may use modules to its left).

## Core Contracts

- All inputs must be `numpy.ndarray`.
- **Images**: shape `(H, W)` or `(H, W, C)` with `C ∈ {1,2,3,4}`. Output dtype is `np.uint8`.
- **Color order: RGB.** Color images are assumed to be in RGB channel order. Not enforced at runtime — it is a documented contract. Callers reading with `cv2.imread` must convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` first.
- **Channel-count semantics**: C=1 grayscale; C=2 grayscale+alpha (alpha ignored); C=3 RGB; C=4 RGBA (alpha ignored).
- **Masks**: shape `(H, W)`, binary values (`bool`, `{0,1}`, `{0,255}`). Output is `np.uint8 ∈ {0, 1}`.
- **BBox**: `(wmin, hmin, wmax, hmax)` — width-first, min-inclusive, max-exclusive.
- Empty-mask functions (`to_roi_box`, `get_box_from_mask`, `get_box_from_coords`, `get_centroid`) return `None`.

## Error Policy

- `TypeError` — wrong input type (non-ndarray).
- `ValueError` — invalid shape, params, mask values, or bbox.
- `RuntimeError` — wrapped lower-level failures (cv2/internal).

## Testing

Tests are contract-focused, organized per module in `tests/{image,mask,spatial,fusion}/`. They verify input validation, output contracts (dtype, shape, binary values), error handling, and edge cases.
