# Web Demo

Flask server that exposes the SwinSeg forgery-localization model as an upload-and-visualize web app, with an optional vision-LLM forensic explanation powered by "Pixy".

## Setup

From the repo root:

```bash
python3 -m venv web/venv
source web/venv/bin/activate
pip install -r web/requirements.txt
```

A trained checkpoint is expected at `model/checkpoints/best_seg.pth`. Without it the server still starts but shows a warning banner and produces meaningless heatmaps — train first with `python model/v4_train.py`.

## Run

```bash
source web/venv/bin/activate
python web/app.py
```

Opens at http://localhost:8000.

## Configuration

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | HTTP listen port |
| `SWINSEG_CKPT` | `model/checkpoints/best_seg.pth` | Model checkpoint path |

### LLM credentials

Stored in `web/config.json` (gitignored). Editable at runtime via the **Settings** panel in the UI without restarting the server.

| Field | Default |
|---|---|
| `api_key` | _(empty)_ |
| `base_url` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `model` | `qwen3.6-plus` |

Upload limits: 10 MB max, accepted MIME types: `image/jpeg`, `image/png`, `image/webp`, `image/bmp`.

## Device selection

```
CUDA → MPS (Apple Silicon) → CPU
```

Picked automatically at startup. No configuration required.

## How it works

### Inference pipeline

1. Upload is validated (MIME type + size), then decoded with Pillow.
2. [app.py](app.py) resizes the image to 512×512, computes a log-normalised grayscale DCT channel, and concatenates it with ImageNet-normalised RGB to form a 4-channel tensor — matching the training pipeline exactly.
3. SwinSeg (imported from `model/v4_train.py`) produces a per-pixel fake-probability map via sigmoid.
4. Three visuals are rendered and returned inline as base64 PNG: the original (512×512), a JET-colormap heatmap, and a blended overlay (55% original / 45% heatmap).
5. Metrics computed: `max_score`, `mean_score`, `forged_ratio` (fraction of pixels > 0.5).

### Concurrency

A `threading.Lock` (`_model_lock`) serialises GPU/MPS access inside `infer()`. Flask runs in threaded mode so preprocessing, image encoding, and LLM calls overlap across concurrent requests while inference is queued.

Results are held in an in-memory LRU cache (10 entries). Nothing is persisted to disk.

### LLM explanation (`/explain`)

Triggered by the **Ask Pixy** button after inference. The server calls a DashScope-hosted vision model (OpenAI-compatible API) with:

- The original image (base64 JPEG, 512×512)
- The heatmap overlay (base64 JPEG, 512×512)
- Numeric stats: `mean_score` and `forged_ratio`

The model responds as **Pixy** — a forensic fairy spirit persona specialised in anime/pixiv-style digital illustrations. Her tone adapts to the verdict: delighted for clean hand-drawn art, smug when catching a small fake, cautious for ambiguous results, indignant for heavily AI-generated images.

> **Domain note:** Pixy is trained exclusively on anime/pixiv-style illustrations. Images outside this domain (photos, realistic art, 3D renders, etc.) trigger a `[STYLE_WARNING]` prefix in the response, surfaced as a banner in the UI.

LLM responses are cached per image SHA-256 (up to 50 entries) to avoid redundant API calls when the same image is re-uploaded.

### API routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Home page |
| `POST` | `/` | Upload image, run inference, render result |
| `POST` | `/explain` | Fetch Pixy's forensic explanation (JSON) |
| `POST` | `/settings` | Update LLM credentials |
| `GET` | `/health` | Health check: `{status, device, checkpoint}` |

## Concurrent load test

```bash
source web/venv/bin/activate
python web/test_concurrent.py
```

Spins up a local test server with a mocked model (no checkpoint needed), fires requests at 1 / 5 / 10 concurrent workers, and reports latency percentiles and throughput. Also runnable via pytest:

```bash
python -m pytest
```
