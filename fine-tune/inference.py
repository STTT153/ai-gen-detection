"""
Inference with the fine-tuned LoRA adapter.

Usage:
    python inference.py --original image.jpg --heatmap heatmap.jpg --prob 0.87
    python inference.py --original image.jpg --heatmap heatmap.jpg --prob 0.87 \
                        --adapter output/qwen-vl-lora
"""

import argparse

import torch


def _import_model_class():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


def load_model(base_model: str, adapter_path: str | None):
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig

    ModelClass = _import_model_class()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model: {base_model}")
    model = ModelClass.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    processor = AutoProcessor.from_pretrained(
        adapter_path if adapter_path else base_model,
        max_pixels=256 * 28 * 28,
    )
    return model, processor


def explain(model, processor, original_path: str, heatmap_path: str, probability: float) -> str:
    from qwen_vl_utils import process_vision_info

    label = "是" if probability > 0.5 else "不是"
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名AI图像检测专家。你将收到两张图像：第一张是待检测的原始图像，"
                "第二张是AI生成概率热力图（颜色越暖/越红表示该区域越可能是AI生成的）。"
                "请根据热力图分布和图像特征，详细解释检测结果。"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": original_path},
                {"type": "image", "image": heatmap_path},
                {
                    "type": "text",
                    "text": (
                        f"整体AI生成概率：{probability:.1%}\n\n"
                        f"请分析以上两张图像，解释为什么模型判断该图像{label}AI生成的，"
                        f"并结合热力图指出关键特征区域。"
                    ),
                },
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    # Only decode the newly generated tokens
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Path to original image")
    parser.add_argument("--heatmap",  required=True, help="Path to heatmap image")
    parser.add_argument("--prob",     required=True, type=float, help="AI probability (0-1)")
    parser.add_argument("--adapter",  default=None,  help="Path to LoRA adapter dir (optional)")
    parser.add_argument(
        "--base-model",
        default="models/Qwen2.5-VL-7B-Instruct",
        help="Base model name or path",
    )
    args = parser.parse_args()

    model, processor = load_model(args.base_model, args.adapter)
    result = explain(model, processor, args.original, args.heatmap, args.prob)

    print("\n" + "=" * 60)
    print("检测结果解释：")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
