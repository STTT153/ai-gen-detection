# Image Style Transfer and Block Swap Pipeline

A two-stage pipeline for AI-powered style transfer with controllable blending.

## Overview

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ Input Image │ ──> │ Style Transfer       │ ──> │ Original +      │
│             │     │ (ControlNet + SD)    │     │ Stylized Images │
└─────────────┘     └──────────────────────┘     └────────┬────────┘
                                                          │
                                                          v
┌─────────────────────────────────────────────────────────────────┐
│                      Block Swap                                 │
│         (N×N grid, swap m blocks between images)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              v
              ┌───────────────┼───────────────┐
              v               v               v
       swapped1.png    swapped2.png      mask.png
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Run the Full Pipeline

```bash
python pipeline.py <input_image> [options]
```

#### Example

```bash
# Basic usage with default style prompt
python pipeline.py images/base1.jpg

# Custom style prompt
python pipeline.py images/base1.jpg -p "watercolor painting style, soft colors, artistic"

# Control block swap parameters
python pipeline.py images/base1.jpg -n 8 -m 10 -s 42

# Clustered swap (organic, irregular regions)
python pipeline.py images/base1.jpg -n 8 -m 10 -c 3
```

### Output Files

The pipeline generates:

| File | Description |
|------|-------------|
| `{name}_original_resized.png` | Input image resized to 512×512 |
| `{name}_stylized.png` | AI-generated stylized version |
| `{name}_swapped1.png` | First blended result |
| `{name}_swapped2.png` | Second blended result |
| `{name}_mask.png` | Swap mask (white = swapped regions) |

---

## Pipeline Components

### 1. Style Transfer (`style_transfer.py`)

Uses Stable Diffusion with ControlNet (Canny edge detection) to preserve composition while changing art style.

```bash
python style_transfer.py <input_image> [options]
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --prompt` | "anime style..." | Target style description |
| `--negative-prompt` | "blurry, low quality..." | Quality filter |
| `-o, --output` | `<input>_stylized.png` | Output path |
| `--no-controlnet` | False | Use simple img2img instead |
| `--strength` | 0.8 | Transformation strength (0.0-1.0) |
| `--steps` | 30 | Inference steps |
| `--guidance` | 7.5 | CFG scale |
| `--device` | auto | `cuda` or `cpu` |

#### Example

```bash
# Generate oil painting style
python style_transfer.py input.jpg -p "oil painting, thick brushstrokes, classical art"

# High-strength transformation
python style_transfer.py input.jpg --strength 0.95 --guidance 9
```

### 2. Block Swap (`block_swap.py`)

Splits two images into an N×N grid, randomly swaps m blocks, and generates outputs.

```bash
python block_swap.py <img1> <img2> [options]
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n` | 4 | Grid size (N×N blocks) |
| `-m` | 3 | Number of blocks to swap |
| `-c, --clusters` | None | Number of spatial clusters |
| `-s, --seed` | None | Random seed |
| `-o, --output` | `output` | Output filename prefix |

#### Examples

```bash
# Default: 4×4 grid, swap 3 blocks
python block_swap.py img1.jpg img2.jpg

# 8×8 grid, swap 10 blocks
python block_swap.py img1.jpg img2.jpg -n 8 -m 10

# Clustered mode: organic, connected regions
python block_swap.py img1.jpg img2.jpg -n 8 -m 10 -c 3 -s 42
```

---

## Pipeline Arguments Reference

| Argument | Component | Description |
|----------|-----------|-------------|
| `input` | pipeline.py | Input image path |
| `-o, --output-dir` | pipeline.py | Output directory |
| `-p, --prompt` | style_transfer | Target style prompt |
| `--negative-prompt` | style_transfer | Negative prompt |
| `--no-controlnet` | style_transfer | Disable ControlNet |
| `--strength` | style_transfer | Style transfer strength |
| `--steps` | style_transfer | Inference steps |
| `--guidance` | style_transfer | CFG scale |
| `-n, --grid-size` | block_swap | Grid size N |
| `-m, --swap-count` | block_swap | Blocks to swap |
| `-c, --clusters` | block_swap | Cluster count |
| `-s, --seed` | block_swap | Random seed |

---

## Tips

1. **Preserve composition**: Always use ControlNet (default) for best structure preservation.
2. **Control blending**: Adjust `-n` and `-m` in block_swap to control how much of the original vs. stylized image appears.
3. **Clustered swaps**: Use `-c` for more organic, connected swapped regions instead of scattered blocks.
4. **Reproducibility**: Use `-s` (seed) to reproduce exact results.

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- diffusers
- opencv-python
- pillow
- numpy

See `requirements.txt` for full list.
