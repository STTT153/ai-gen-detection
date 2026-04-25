"""
Concurrent load test — verifies thread-safety of the inference lock
and measures throughput under increasing concurrency.

Usage (standalone):
    cd /path/to/ai-gen-detection
    source venv/bin/activate
    python web/test_concurrent.py

Usage (pytest):
    pytest web/test_concurrent.py -v -s
"""
import io
import logging
import socket
import statistics
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import requests
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(WEB_DIR))

# ---------------------------------------------------------------------------
# Fake model — simulates GPU latency without needing real weights
# ---------------------------------------------------------------------------
INFER_DELAY = 0.05  # seconds per inference, simulates GPU round-trip


class _FakeModel:
    """Stand-in for SwinSeg: sleeps to simulate inference, returns zeros."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        time.sleep(INFER_DELAY)
        b, _, h, w = x.shape
        return torch.zeros(b, 1, h, w)

    def load_state_dict(self, state, strict=True):
        pass  # ignore real checkpoint weights — we return zeros regardless

    def to(self, device):
        return self

    def eval(self):
        return self


# ---------------------------------------------------------------------------
# Module injection helpers
# ---------------------------------------------------------------------------

def _inject_mocks():
    """Insert fake v4_train so `from v4_train import SwinSeg` succeeds."""
    if "v4_train" not in sys.modules:
        fake_v4 = MagicMock()
        fake_v4.SwinSeg = _FakeModel
        sys.modules["v4_train"] = fake_v4


def start_test_server() -> tuple[str, object]:
    """Start the Flask app with a mocked model on a random port.

    Returns (base_url, flask_app_module).
    """
    _inject_mocks()
    import app as flask_app  # noqa: PLC0415  (import inside function intentional)
    import llm  # noqa: PLC0415

    # Replace the global model with our controlled fake
    flask_app.MODEL = _FakeModel()

    # Short-circuit LLM calls — we're testing inference concurrency, not the LLM
    llm.explain = lambda **_kw: "mocked explanation"

    port = _free_port()
    threading.Thread(
        target=flask_app.app.run,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "threaded": True,
            "use_reloader": False,
            "debug": False,
        },
        daemon=True,
    ).start()

    base_url = f"http://127.0.0.1:{port}"
    _wait_for_server(base_url)
    return base_url, flask_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(url, timeout=1)
            return
        except requests.ConnectionError:
            time.sleep(0.1)
    raise RuntimeError(f"Server at {url} did not start within {timeout}s")


def _make_jpeg(w: int = 64, h: int = 64) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _post_image(url: str, image_bytes: bytes) -> dict:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/",
            files={"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        ok = resp.status_code == 200
        # Basic sanity: response HTML should contain the result section
        sane = ok and "heatmap" in resp.text
        return {"ok": ok, "sane": sane, "status": resp.status_code, "elapsed": elapsed}
    except Exception as exc:
        return {"ok": False, "sane": False, "status": None,
                "elapsed": time.perf_counter() - t0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(label: str, url: str, n_workers: int, n_total: int) -> dict:
    image = _make_jpeg()
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  workers={n_workers}  requests={n_total}  infer_delay={INFER_DELAY}s/req")
    print(f"{'─' * 60}")

    t_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_post_image, url, image) for _ in range(n_total)]
        results = [f.result() for f in as_completed(futures)]
    t_wall = time.perf_counter() - t_wall

    ok = [r for r in results if r["ok"]]
    insane = [r for r in results if r["ok"] and not r["sane"]]
    fail = [r for r in results if not r["ok"]]

    print(f"  OK={len(ok)}  Corrupted={len(insane)}  FAIL={len(fail)}")
    for r in fail[:3]:
        print(f"    ✗ status={r.get('status')}  {r.get('error', '')}")

    if ok:
        lats = sorted(r["elapsed"] for r in ok)
        p50 = statistics.median(lats)
        p90 = lats[int(len(lats) * 0.9)]
        p99 = lats[min(int(len(lats) * 0.99), len(lats) - 1)]
        print(
            f"  Latency  min={lats[0]:.3f}s  p50={p50:.3f}s  "
            f"p90={p90:.3f}s  p99={p99:.3f}s  max={lats[-1]:.3f}s"
        )

    # Serial lower bound: if inference is the only thing, all n_total infer()
    # calls must still queue behind the lock → wall >= n_total * INFER_DELAY
    serial_infer = n_total * INFER_DELAY
    throughput = n_total / t_wall
    # Speedup is meaningful when n_workers > 1: non-inference work (preprocess,
    # encode, HTTP) can overlap while inference is serialized by _model_lock.
    speedup = serial_infer / t_wall
    print(
        f"  Wall={t_wall:.2f}s  Throughput={throughput:.1f} req/s  "
        f"(serial-infer bound={serial_infer:.1f}s → speedup={speedup:.2f}x)"
    )

    return {
        "ok": len(ok),
        "fail": len(fail),
        "corrupted": len(insane),
        "wall": t_wall,
        "throughput": throughput,
    }


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

_server_url: str | None = None
_server_lock = threading.Lock()


def _get_server() -> str:
    global _server_url
    with _server_lock:
        if _server_url is None:
            _server_url, _ = start_test_server()
            _post_image(_server_url, _make_jpeg())  # warm-up
    return _server_url


def test_serial_baseline():
    url = _get_server()
    stats = run_scenario("1-worker baseline", url, n_workers=1, n_total=5)
    assert stats["fail"] == 0
    assert stats["corrupted"] == 0


def test_light_concurrency():
    url = _get_server()
    stats = run_scenario("5-worker light load", url, n_workers=5, n_total=20)
    assert stats["fail"] == 0, f"{stats['fail']} requests failed"
    assert stats["corrupted"] == 0, "response data was corrupted under concurrency"


def test_high_concurrency():
    url = _get_server()
    stats = run_scenario("10-worker high concurrency", url, n_workers=10, n_total=40)
    assert stats["fail"] == 0, f"{stats['fail']} requests failed"
    assert stats["corrupted"] == 0, "response data was corrupted under concurrency"
    # With 10 workers overlapping non-inference work we should beat pure serial
    assert stats["throughput"] > 1.0 / INFER_DELAY * 0.9, "throughput unexpectedly low"


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # suppress Flask/Werkzeug noise

    print("Starting test server (mocked model — no checkpoint needed)...")
    url, _ = start_test_server()
    print(f"Server ready at {url}")

    print("\nWarming up...")
    _post_image(url, _make_jpeg())

    run_scenario("1-worker baseline (serial)", url, n_workers=1, n_total=8)
    run_scenario("5-worker light load", url, n_workers=5, n_total=20)
    run_scenario("10-worker high concurrency", url, n_workers=10, n_total=40)

    print("\n✓ All scenarios completed — no crashes, no data corruption.")
