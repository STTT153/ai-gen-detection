# Web Demo

Flask server that exposes the SwinSeg forgery-localization model as an upload-and-visualize web app.

## Setup

From the repo root:

```bash
python3 -m venv venv          # if not already present
source venv/bin/activate
pip install -r web/requirements.txt
```

A trained checkpoint at `AI_Detection/Mask_Model/best_seg.pth` is expected. Without it, the server still runs but shows a warning banner and produces meaningless heatmaps — train first with `python AI_Detection/mask_train_model.py`.

## Run

```bash
source venv/bin/activate       # from repo root
python web/app.py
```

Open http://localhost:5000, upload an image, get back three views: original (resized to 224×224), jet heatmap, and blended overlay. Header reports max/mean sigmoid score and forged-area percentage at threshold 0.5.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | HTTP port |
| `SWINSEG_CKPT` | `AI_Detection/Mask_Model/best_seg.pth` | Model checkpoint path |

Upload limits: 10 MB, MIME types `image/jpeg|png|webp|bmp`.

## How it works

[app.py](app.py) imports `SwinSeg` directly from [../AI_Detection/mask_train_model.py](../AI_Detection/mask_train_model.py) and replicates `MaskDataset.__getitem__` preprocessing (224-resize, DCT channel, ImageNet normalization, 4-channel concat). This keeps train and serve paths in lockstep — if the preprocessing changes, update both sides.

Inference is stateless and synchronous; uploads stream through memory and results render as base64 PNGs inline, so nothing is persisted to disk.

## Caveat

The model was trained on 224×224 composited portraits (AI→Real and Real→AI). Arbitrary photos outside that distribution will produce unreliable maps even with a good checkpoint — that's a domain limit of the training data, not a server bug.
