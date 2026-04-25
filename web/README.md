# Web Demo

Flask server that exposes the SwinSeg forgery-localization model as an upload-and-visualize web app.

## Setup

From the repo root:

```bash
python3 -m venv web/venv
source web/venv/bin/activate
pip install -r web/requirements.txt
```

A trained checkpoint at `model/checkpoints/best_seg_ft.pth` is expected. Without it the server still runs but shows a warning banner and produces meaningless heatmaps — train first with `python model/mask_train_model.py`.

## Run

### Development

```bash
source web/venv/bin/activate
python web/app.py
```

Opens at http://localhost:8000.

### Production (recommended)

Use gunicorn with a single process and a thread pool so the model is loaded once and requests are handled concurrently:

```bash
source web/venv/bin/activate
gunicorn -w 1 -k gthread --threads 4 -b 0.0.0.0:8000 --chdir web app:app
```

| Flag | Value | Reason |
|---|---|---|
| `-w 1` | 1 worker process | Model loaded once; avoids duplicating GPU memory |
| `-k gthread --threads 4` | gthread worker, 4 threads | Concurrent requests; I/O (LLM calls) overlaps while inference is serialized |
| `--chdir web` | change into web/ before starting | Resolves `app:app` and `config.json` correctly |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | HTTP port (dev server only) |
| `SWINSEG_CKPT` | `model/checkpoints/best_seg_ft.pth` | Model checkpoint path |

LLM credentials (API key, endpoint, model name) are stored in `web/config.json` (gitignored) and can be set via the Settings panel in the UI.

Upload limits: 10 MB, MIME types `image/jpeg|png|webp|bmp`.

## Device selection

The server picks the best available compute device automatically:

```
CUDA → MPS (Apple Silicon) → CPU
```

On an M-series Mac the model runs on MPS (Metal), which uses the built-in GPU cores and is significantly faster than CPU. No configuration required.

## How it works

[app.py](app.py) imports `SwinSeg` from `model/v4_train.py` and runs the same preprocessing pipeline used during training: resize to 512×512, compute a log-normalised DCT channel, concatenate with ImageNet-normalised RGB to form a 4-channel tensor.

A `threading.Lock` (`_model_lock`) serialises access to the model so concurrent requests don't race on the GPU/MPS context. Preprocessing, image encoding, and LLM calls run in parallel across threads.

Uploads stream through memory; results render as base64 PNGs inline — nothing is persisted to disk.

## Running the concurrent load test

```bash
source web/venv/bin/activate
python web/test_concurrent.py
```

Starts a local test server with a mocked model (no checkpoint needed), fires requests at 1 / 5 / 10 concurrent workers, and reports latency percentiles and throughput. Also runnable via pytest (configured in `pyproject.toml` at the repo root):

```bash
source web/venv/bin/activate
python -m pytest
```

## Caveat

The model was trained on 512×512 composited portraits (AI→Real and Real→AI). Arbitrary photos outside that distribution will produce unreliable maps even with a good checkpoint — that is a domain limit of the training data, not a server bug.
