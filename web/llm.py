"""Client for a DashScope-compatible (OpenAI format) vision LLM endpoint.

Sends the original image and heatmap overlay (base64 JPEG) together with
SwinSeg's detection statistics to a multimodal model for forensic analysis.
"""
from __future__ import annotations

import openai

SYSTEM_PROMPT = """You are a forensic analyst for AI art fraud detection. The tool you are assisting is SwinSeg, a pixel-level segmentation model that detects AI-generated regions in digital illustrations.

DOMAIN: SwinSeg was trained exclusively on anime/pixiv-style digital illustrations. It is NOT reliable on photographs, oil paintings, sketches, 3D renders, or other non-anime styles.

WHAT YOU RECEIVE:
1. The original image (512×512)
2. A heatmap overlay (JET colormap) blended onto the original:
   blue = likely hand-drawn; green/yellow/red = increasingly likely AI-generated
3. Two key statistics: mean score and forged_ratio (% of pixels above 0.5 threshold)

STYLE CHECK (silent):
If the image is NOT anime/pixiv-style digital illustration (e.g. photograph, realistic painting, sketch, 3D render, western cartoon), output this token on the very first line followed by one sentence explaining why, then a blank line:
[STYLE_WARNING] <reason>

If the style matches, output nothing about the style check — start directly with the forensic description.

FORENSIC DESCRIPTION:
Based on the heatmap, mean score, and forged_ratio, write a concise forensic description:
- Where in the image are the AI-suspected regions (location, shape, rough size)?
- How severe is the forgery (localized detail vs. large area vs. whole image)?
- How confident is the detection, based on the score level and how clearly the heatmap highlights a region?

Do NOT classify into named categories. Describe what you observe in plain language.
Keep the forensic description under 150 words."""


def _user_text(mean_score: float, forged_ratio: float) -> str:
    return (
        "Image 1 is the original artwork. Image 2 is the heatmap overlay "
        "(blue=hand-drawn, red/yellow=AI-generated).\n\n"
        f"Detection statistics:\n"
        f"- mean score: {mean_score:.3f}\n"
        f"- forged_ratio (pixels > 0.5): {forged_ratio * 100:.1f}%\n\n"
        "Describe what you observe about the AI-suspected regions."
    )


_STYLE_WARNING_TOKEN = "[STYLE_WARNING]"


def explain(
    *,
    api_key: str,
    base_url: str,
    model: str,
    mean_score: float,
    forged_ratio: float,
    orig_b64: str,
    overlay_b64: str,
    max_tokens: int = 600,
    timeout: float = 60.0,
) -> tuple[str, str | None]:
    """Call the vision LLM with the original image, heatmap overlay, and stats.

    Returns (explanation, style_warning). style_warning is None if the image
    style matches the training domain.
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    msg = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{overlay_b64}"},
                    },
                    {"type": "text", "text": _user_text(mean_score, forged_ratio)},
                ],
            },
        ],
    )
    raw = (msg.choices[0].message.content or "").strip()

    if raw.startswith(_STYLE_WARNING_TOKEN):
        first_line, _, rest = raw.partition("\n")
        style_warning = first_line[len(_STYLE_WARNING_TOKEN):].strip()
        return rest.strip(), style_warning

    return raw, None
