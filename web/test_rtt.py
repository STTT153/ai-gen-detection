"""RTT test for the configured LLM relay.

Tests three layers:
  1. TCP connect time to the relay host
  2. Minimal text-only API call (no images)
  3. Vision API call with two small JPEG images (matches production payload)

Usage:
    python web/test_rtt.py [--rounds N]
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import time
import urllib.parse
from pathlib import Path

import anthropic
import numpy as np
from PIL import Image
import io

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CONFIG = {
    "api_key": "",
    "base_url": "https://www.fucheers.top",
    "model": "claude-sonnet-4-6",
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
    return dict(DEFAULT_CONFIG)


def tcp_rtt(host: str, port: int, rounds: int) -> list[float]:
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        s = socket.create_connection((host, port), timeout=10)
        s.close()
        times.append(time.perf_counter() - t0)
    return times


def _small_jpeg_b64(size: int = 64) -> str:
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def api_text_rtt(client: anthropic.Anthropic, model: str, rounds: int) -> list[float]:
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        )
        times.append(time.perf_counter() - t0)
    return times


def api_vision_rtt(client: anthropic.Anthropic, model: str, rounds: int) -> list[float]:
    img_b64 = _small_jpeg_b64(64)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        {"type": "text", "text": "Describe the image in one word."},
    ]
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": content}],
        )
        times.append(time.perf_counter() - t0)
    return times


def stats(times: list[float]) -> str:
    a = sorted(times)
    mean = sum(a) / len(a)
    return f"min={a[0]*1000:.0f}ms  avg={mean*1000:.0f}ms  max={a[-1]*1000:.0f}ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per test (default: 3)")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.get("api_key"):
        print("ERROR: no api_key in config.json")
        return

    parsed = urllib.parse.urlparse(cfg["base_url"])
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    model = cfg["model"]

    print(f"Relay : {cfg['base_url']}")
    print(f"Model : {model}")
    print(f"Rounds: {args.rounds}")
    print()

    print(f"[1] TCP connect to {host}:{port}")
    t = tcp_rtt(host, port, args.rounds)
    print(f"    {stats(t)}")
    print()

    client = anthropic.Anthropic(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=60.0)

    print("[2] Text-only API call (max_tokens=16)")
    t = api_text_rtt(client, model, args.rounds)
    print(f"    {stats(t)}")
    print()

    print("[3] Vision API call (2× 64×64 JPEG, max_tokens=16)")
    t = api_vision_rtt(client, model, args.rounds)
    print(f"    {stats(t)}")


if __name__ == "__main__":
    main()
