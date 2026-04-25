"""
Batch-generate explanation labels using Claude claude-opus-4-7 (vision).

Workflow:
  1. Your AI detection model processes images → produces heatmaps + probabilities
  2. This script feeds (original, heatmap, probability) to Claude → explanation text
  3. Review/edit the output JSON, then split into train.json / val.json for fine-tuning

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_labels.py --input unlabeled.json --output labeled.json

unlabeled.json format (no "explanation" field needed):
[
    {
        "original_image": "images/img001.jpg",
        "heatmap_image":  "heatmaps/img001_heatmap.jpg",
        "probability":    0.87
    },
    ...
]
"""

import argparse
import base64
import json
import time
from pathlib import Path


SYSTEM_PROMPT = """你是一名AI图像检测专家。
你将收到两张图像（第一张：原始图像，第二张：AI生成概率热力图，颜色越暖/越红=概率越高）以及整体AI生成概率。
请用中文写出200-400字的专业分析，内容包括：
1. 总体判断（是/否AI生成）及置信度
2. 热力图中的高概率区域位置及其视觉特征
3. 具体的AI生成痕迹（如：眼睛对称性异常、手指结构错误、纹理过于均匀、边缘伪影等）
4. 为什么这些特征是AI生成的标志

只输出分析文本，不要输出其他任何内容。"""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def image_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")


def generate_explanation(client, original: str, heatmap: str, probability: float) -> str:
    label = "是" if probability > 0.5 else "不是"
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type(original),
                            "data": encode_image(original),
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type(heatmap),
                            "data": encode_image(heatmap),
                        },
                    },
                    {
                        "type": "text",
                        "text": f"整体AI生成概率：{probability:.1%}（模型判断该图像{label}AI生成的）",
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Unlabeled JSON file")
    parser.add_argument("--output", required=True, help="Output labeled JSON file")
    parser.add_argument("--delay",  default=1.0, type=float,
                        help="Seconds between API calls (avoid rate limits)")
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    # Load partial results if output already exists (resume support)
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            results = json.load(f)
        done_keys = {(r["original_image"], r["heatmap_image"]) for r in results}
    else:
        results = []
        done_keys = set()

    for i, item in enumerate(data):
        key = (item["original_image"], item["heatmap_image"])
        if key in done_keys:
            print(f"[{i+1}/{len(data)}] Skipping (already done): {item['original_image']}")
            continue

        print(f"[{i+1}/{len(data)}] Generating: {item['original_image']}")
        try:
            explanation = generate_explanation(
                client,
                item["original_image"],
                item["heatmap_image"],
                float(item["probability"]),
            )
            result = {**item, "explanation": explanation}
            results.append(result)
            done_keys.add(key)

            # Save after every item (crash recovery)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"  ✓ {len(explanation)} chars")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        if i < len(data) - 1:
            time.sleep(args.delay)

    print(f"\nDone. {len(results)}/{len(data)} examples saved to {args.output}")


if __name__ == "__main__":
    main()
