"""Lightweight Flask server for AI forgery localization demo.

Accepts an image upload, runs SwinSeg inference, returns heatmap + overlay,
and optionally forwards the result to an Anthropic-compatible LLM (Bailian)
for a natural-language explanation.
"""
import base64
import io
import json
import logging
import os
import sys
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize

ROOT = Path(__file__).resolve().parent.parent
AI_DETECTION_DIR = ROOT / "model"
CHECKPOINT_PATH = Path(
    os.environ.get("SWINSEG_CKPT", AI_DETECTION_DIR / "checkpoints" / "best_seg.pth")
)
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

sys.path.insert(0, str(AI_DETECTION_DIR))
from v4_train import SwinSeg  # noqa: E402
import llm  # noqa: E402

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://www.fucheers.top",
    "model": "claude-sonnet-4-6",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
        except json.JSONDecodeError:
            log.warning("config.json is invalid JSON, using defaults")
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 10:
        return "•" * len(key)
    return f"{key[:6]}…{key[-4:]}"


def load_model():
    if DEVICE.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = SwinSeg()
    checkpoint_loaded = False
    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        checkpoint_loaded = True
        log.info(f"loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        log.warning(f"no checkpoint at {CHECKPOINT_PATH} — predictions will be meaningless")
    model.to(DEVICE).eval()
    return model, checkpoint_loaded


MODEL, HAS_CHECKPOINT = load_model()
_model_lock = threading.Lock()

_result_cache: OrderedDict[str, dict] = OrderedDict()
_CACHE_MAX = 10


def _cache_put(data: dict) -> str:
    rid = str(uuid.uuid4())
    _result_cache[rid] = data
    if len(_result_cache) > _CACHE_MAX:
        _result_cache.popitem(last=False)
    return rid


def preprocess(pil_img: Image.Image):
    img_rgb = np.array(pil_img.convert("RGB"))
    img_rgb = cv2.resize(img_rgb, (512, 512))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct = np.log(np.abs(dct) + 1e-6)
    dct = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img_t = transforms.ToTensor()(Image.fromarray(img_rgb))
    img_t = normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dct_t = torch.from_numpy(dct).float().unsqueeze(0) / 255.0

    x = torch.cat([img_t, dct_t], dim=0).unsqueeze(0)
    return x, img_rgb


@torch.no_grad()
def infer(x: torch.Tensor) -> np.ndarray:
    with _model_lock:
        x = x.to(DEVICE)
        logits = MODEL(x)
        return torch.sigmoid(logits).squeeze().cpu().numpy()


def render_visuals(img_rgb: np.ndarray, prob: np.ndarray):
    heatmap_u8 = (prob * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_rgb, 0.45, 0)

    binary = (prob > 0.5).astype(np.float32)
    forged_ratio = float(binary.mean())

    return {
        "original": _png_b64(img_rgb),
        "heatmap": _png_b64(heatmap_rgb),
        "overlay": _png_b64(overlay),
        "orig_jpeg": _jpeg_b64(img_rgb),
        "overlay_jpeg": _jpeg_b64(overlay),
        "forged_ratio": forged_ratio,
        "max_score": float(prob.max()),
        "mean_score": float(prob.mean()),
    }


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jpeg_b64(rgb: np.ndarray, quality: int = 75) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _view_ctx(cfg: dict, **kw):
    """Common template context; merges settings visibility with per-request data."""
    return {
        "has_checkpoint": HAS_CHECKPOINT,
        "settings": {
            "api_key_masked": mask_key(cfg.get("api_key", "")),
            "api_key_set": bool(cfg.get("api_key")),
            "base_url": cfg.get("base_url", ""),
            "model": cfg.get("model", ""),
        },
        "result": None,
        "result_id": None,
        "error": None,
        "settings_saved": False,
        **kw,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    cfg = load_config()
    saved = bool(request.args.get("saved"))

    if request.method == "GET":
        return render_template("index.html", **_view_ctx(cfg, settings_saved=saved))

    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", **_view_ctx(cfg, error="Please select an image.")), 400
    if file.mimetype not in ALLOWED_MIME:
        return render_template("index.html", **_view_ctx(cfg, error=f"Unsupported image type: {file.mimetype}")), 400

    try:
        pil_img = Image.open(file.stream)
        pil_img.load()
    except Exception as e:
        return render_template("index.html", **_view_ctx(cfg, error=f"Could not read image: {e}")), 400

    x, img_rgb = preprocess(pil_img)
    prob = infer(x)
    result = render_visuals(img_rgb, prob)

    result_id = _cache_put({
        "mean_score": result["mean_score"],
        "forged_ratio": result["forged_ratio"],
        "orig_b64": result["orig_jpeg"],
        "overlay_b64": result["overlay_jpeg"],
    })

    return render_template("index.html", **_view_ctx(cfg, result=result, result_id=result_id))


@app.route("/explain", methods=["POST"])
def explain_route():
    from flask import jsonify
    cfg = load_config()
    if not cfg.get("api_key"):
        return jsonify({"error": "API key not configured"}), 400

    data = request.get_json(silent=True) or {}
    cached = _result_cache.get(data.get("result_id"))
    if not cached:
        return jsonify({"error": "Result expired, please re-upload the image"}), 404

    try:
        explanation, style_warning = llm.explain(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
            mean_score=cached["mean_score"],
            forged_ratio=cached["forged_ratio"],
            orig_b64=cached["orig_b64"],
            overlay_b64=cached["overlay_b64"],
        )
        return jsonify({"explanation": explanation, "style_warning": style_warning})
    except Exception as e:
        log.exception("LLM call failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/settings", methods=["POST"])
def update_settings():
    cfg = load_config()
    form = request.form

    new_key = form.get("api_key", "").strip()
    # Empty key field means "keep existing" unless explicit clear is sent
    if new_key:
        cfg["api_key"] = new_key
    elif form.get("clear_key"):
        cfg["api_key"] = ""

    base_url = form.get("base_url", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    model = form.get("model", "").strip()
    if model:
        cfg["model"] = model

    save_config(cfg)
    return redirect(url_for("index") + "?saved=1")


@app.errorhandler(413)
def too_large(_):
    cfg = load_config()
    return render_template("index.html", **_view_ctx(
        cfg, error=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB).",
    )), 413


@app.route("/health")
def health():
    from flask import jsonify
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint": HAS_CHECKPOINT,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
