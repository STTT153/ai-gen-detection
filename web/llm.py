"""Client for an Anthropic-compatible LLM endpoint (e.g. Bailian at
coding.dashscope.aliyuncs.com). Forwards SwinSeg's detection stats to the
LLM and asks for a natural-language explanation.

The Bailian `/apps/anthropic` endpoint currently only whitelists text-only
models (qwen3-coder-plus), so we send spatial summary statistics rather
than image blocks.
"""
from __future__ import annotations

import anthropic
import numpy as np

SYSTEM_PROMPT = """You are a forensic analyst interpreting output from an AI image-forgery detection model called SwinSeg.

The model segments images into "real" vs "AI-generated" regions. It was trained on two forgery types:
- AI→Real: an AI-generated person composited onto a real background. Expect forged area to match a human silhouette (roughly 15-50% of the frame, usually centered or slightly off-center).
- Real→AI: a real person composited onto an AI-generated background. Expect forged area to be large (>50%) and peripheral, wrapping around the person-shaped hole.

You cannot see the image itself. Instead you receive numeric summary statistics over the model's pixel-wise fake-probability map. Use them to:
1. Infer which forgery scenario is most likely (AI→Real vs Real→AI) and why.
2. Describe where the flagged region sits in the frame (centered, peripheral, top, bottom, etc.).
3. State how confident the model is, based on the score distribution.

Be specific, cite the numbers, and keep the answer under 200 words. If the stats are ambiguous, say so."""


def _spatial_stats(prob: np.ndarray, thr: float = 0.5) -> dict:
    """Compute where-in-the-frame summary of the fake mask."""
    h, w = prob.shape
    mask = prob > thr
    total = max(mask.sum(), 1)

    top = mask[: h // 2].sum() / total
    bottom = mask[h // 2 :].sum() / total
    left = mask[:, : w // 2].sum() / total
    right = mask[:, w // 2 :].sum() / total

    # center = inner 50% (h/4..3h/4 × w/4..3w/4)
    ch, cw = h // 4, w // 4
    center = mask[ch : h - ch, cw : w - cw].sum() / total

    # score percentiles
    p50, p90, p99 = np.percentile(prob, [50, 90, 99])

    return {
        "top_frac": float(top),
        "bottom_frac": float(bottom),
        "left_frac": float(left),
        "right_frac": float(right),
        "center_frac": float(center),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
    }


def _user_text(max_score: float, mean_score: float, forged_ratio: float, prob: np.ndarray) -> str:
    s = _spatial_stats(prob)
    return (
        f"SwinSeg per-pixel fake-probability map over a 224×224 image.\n\n"
        f"Score distribution:\n"
        f"- max: {max_score:.3f}\n"
        f"- mean: {mean_score:.3f}\n"
        f"- 50th/90th/99th percentile: {s['p50']:.3f} / {s['p90']:.3f} / {s['p99']:.3f}\n\n"
        f"Thresholded (>0.5) mask:\n"
        f"- fraction of image flagged as fake: {forged_ratio * 100:.1f}%\n"
        f"- of flagged pixels, share in top half / bottom half: {s['top_frac']*100:.0f}% / {s['bottom_frac']*100:.0f}%\n"
        f"- of flagged pixels, share in left half / right half: {s['left_frac']*100:.0f}% / {s['right_frac']*100:.0f}%\n"
        f"- of flagged pixels, share in inner 50% (center): {s['center_frac']*100:.0f}%\n\n"
        f"Given these numbers, which forgery scenario is most likely, where is the flagged region, "
        f"and how confident is the model?"
    )


def explain(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prob: np.ndarray,
    max_score: float,
    mean_score: float,
    forged_ratio: float,
    max_tokens: int = 600,
    timeout: float = 60.0,
) -> str:
    """Call the LLM with summary stats derived from `prob`; return its explanation."""
    client = anthropic.Anthropic(auth_token=api_key, base_url=base_url, timeout=timeout)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_text(max_score, mean_score, forged_ratio, prob)}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip()
