# ai-gen-detection

## Introduction

AI image forgery detection system for identifying AI-generated regions within artworks. The system targets a real-world "cheating" scenario: artists who use AI to generate parts of an image (a background, a figure, a face) while falsely claiming the work is entirely hand-drawn.

The core model, **SwinSeg**, performs pixel-level segmentation to localize forged regions. A Flask web UI wraps inference with a heatmap visualizer and an optional LLM-powered forensic explanation.

Two forgery directions are supported:

## Project Structure

```
ai-gen-detection/
├── model/                          # Model training & evaluation (git submodule)
│   ├── v4_train.py                 # SwinSeg model definition + training script
│   ├── infer.py                    # Standalone inference (single image or folder)
│   ├── checkpoints/                # Saved model weights (gitignored)
│   └── requirements.txt
├── web/                            # Flask web interface
│   ├── app.py                      # Server: inference, heatmap rendering, LLM integration
│   ├── llm.py                      # Anthropic-compatible LLM client
│   ├── gunicorn.conf.py            # Production server config
│   ├── test_concurrent.py          # Concurrent load test
│   ├── config.json                 # API credentials (gitignored)
│   ├── requirements.txt
│   └── templates/index.html        # Dark-themed single-page UI
├── data/                           # Dataset generation pipelines
│   ├── random-ai-substitution/     # Style-transfer + block-swap pipeline
│   ├── person_background_cutmix/   # Cutout & composite helpers
│   └── inpaint_pipeline.py         # SAM + SD inpainting (experimental)
└── pyproject.toml                  # Pytest config
```

## Dataset

The dataset covers six forgery categories to capture the full range of ways an artist might mix AI and hand-drawn content:

| Category | Description |
|---|---|
| **Pure Real** | Entire image is hand-drawn (source: Kaggle pixiv-2020) |
| **Pure AI** | Entire image is AI-generated (`animagine-anything-v5.ipynb`) |
| **Real on AI** | Hand-drawn figure on an AI-generated background |
| **AI on Real** | AI-generated figure on a hand-drawn background |
| **Random AI** | Randomly selected patches replaced with AI content |
| **Specific AI** | Semantically meaningful area (face, hands, clothing) replaced with AI content |

**Data format:** 512×512 PNG images paired with `.npy` mask files (`1` = AI-generated, `0` = real).

Generation tools are in [data/](data/). The `inpaint_pipeline.py` approach was abandoned; see [data/README.md](data/README.md) for details.

## Model Architecture: SwinSeg

SwinSeg is a U-Net-style segmentation model built on a Swin Transformer backbone.

**Input:** 4-channel tensor — RGB (ImageNet-normalized) + 1-channel DCT feature map. The DCT channel exposes JPEG compression artifacts and frequency-domain anomalies left by AI generation, which are invisible in RGB alone.

**Backbone:** `swin_tiny_patch4_window7_224` (pretrained ImageNet), patched to accept 4-channel input and run at 512×512.

**Decoder:** 4-level U-Net decoder with skip connections and bilinear upsampling, producing a 1-channel per-pixel fake-probability logit map at 512×512.

**Loss:** Dice loss + Focal loss (α=0.25, γ=2.0) with 1:1 balanced sampling across forgery directions.

**Training split:** 280 train / 40 val / 90 test samples, balanced per forgery type.

## Web Application

The web UI accepts an image upload and returns three visualizations:

- **Original** — resized to 512×512
- **Heatmap** — JET-colormap over the per-pixel fake-probability map
- **Overlay** — 55% original blended with 45% heatmap for spatial context

Reported metrics include max score, mean score, and `forged_ratio` (percentage of pixels above the 0.5 threshold).

An optional LLM panel (powered by an Anthropic-compatible endpoint) receives only numeric spatial statistics — no image pixels — and returns a natural-language forensic explanation classifying the forgery direction.

### Setup & Running

```bash
# Create virtualenv and install dependencies
python3 -m venv web/venv
source web/venv/bin/activate
pip install -r web/requirements.txt

# Development server (http://localhost:8000)
python web/app.py

# Production server
gunicorn --chdir web --config web/gunicorn.conf.py app:app
```

Optional environment variables:

```bash
PORT=8000                                         # default: 8000
SWINSEG_CKPT=model/checkpoints/best_seg_ft.pth   # path to model checkpoint
```

LLM credentials are configured via the Settings panel in the UI and stored in `web/config.json` (gitignored). Default endpoint: Aliyun Bailian (`qwen3-coder-plus`).

### Device Selection

The server automatically selects the best available backend: **CUDA → MPS (Apple Silicon) → CPU**. No configuration required.

### Concurrency

A `threading.Lock` serialises GPU/MPS access during inference. Preprocessing, image encoding, and LLM calls overlap freely across concurrent requests.

## Model Training & Inference

```bash
# Train (saves checkpoint to model/checkpoints/best_seg_ft.pth)
cd model
python v4_train.py

# Run inference on a single image or folder
python infer.py --ckpt checkpoints/best_seg_ft.pth --input <image_or_dir>
```

Key training hyperparameters:

| Parameter | Value |
|---|---|
| Input resolution | 512×512 |
| Batch size | 8 |
| Epochs | 20 |
| Backbone LR | 5e-6 (AdamW, cosine schedule) |
| Decoder LR | 1e-4 (AdamW, cosine schedule) |
| Loss | Dice + Focal (α=0.25, γ=2.0) |

## Tests

```bash
source web/venv/bin/activate
python -m pytest                 # runs web/test_concurrent.py via pyproject.toml
python web/test_concurrent.py    # standalone with printed throughput report
```

## Reference
