"""
QLoRA fine-tuning for AI image detection explanation task.

Input:  original image + heatmap image + overall AI probability
Output: natural language explanation of the detection result

Memory budget for RTX 4060 8GB:
  - 4-bit model weights:  ~3.5 GB
  - LoRA + optimizer:     ~0.8 GB
  - Activations (with grad checkpoint): ~2.5 GB
  Total:                  ~6.8 GB
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_NAME      = "models/Qwen2.5-VL-7B-Instruct"    # local path after running download_model.py
TRAIN_DATA      = "data/train.json"
VAL_DATA        = "data/val.json"                  # optional; skipped if missing
OUTPUT_DIR      = "output/qwen-vl-lora"

LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

BATCH_SIZE      = 1
GRAD_ACCUM      = 8                                # effective batch size = 8
NUM_EPOCHS      = 3
LR              = 2e-4
MAX_SEQ_LEN     = 1024
# Limit image tokens per image (~256 tokens each at this setting)
MAX_PIXELS      = 256 * 28 * 28
# ──────────────────────────────────────────────────────────────────────────────


def _import_model_class():
    """Try Qwen2.5-VL first, fall back to Qwen2-VL."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration


def _resolve_path(base_dir: Path, p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


class AIDetectionDataset(Dataset):
    """
    Expects a JSON file with a list of objects:
    {
        "original_image": "path/to/original.jpg",
        "heatmap_image":  "path/to/heatmap.jpg",
        "probability":    0.87,            # float 0-1
        "explanation":    "该图像被判定为..."
    }
    Paths can be absolute or relative to the JSON file's directory.
    """

    SYSTEM_PROMPT = (
        "你是一名AI图像检测专家。你将收到两张图像：第一张是待检测的原始图像，"
        "第二张是AI生成概率热力图（颜色越暖/越红表示该区域越可能是AI生成的）。"
        "请根据热力图分布和图像特征，详细解释检测结果。"
    )

    def __init__(self, data_path: str, processor: AutoProcessor, max_length: int):
        self.data_path = Path(data_path)
        self.base_dir = self.data_path.parent
        with open(data_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _build_messages(self, example: dict) -> list:
        orig = _resolve_path(self.base_dir, example["original_image"])
        hmap = _resolve_path(self.base_dir, example["heatmap_image"])
        prob = float(example["probability"])
        label = "是" if prob > 0.5 else "不是"

        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": orig},
                    {"type": "image", "image": hmap},
                    {
                        "type": "text",
                        "text": (
                            f"整体AI生成概率：{prob:.1%}\n\n"
                            f"请分析以上两张图像，解释为什么模型判断该图像{label}AI生成的，"
                            f"并结合热力图指出关键特征区域。"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": example["explanation"],
            },
        ]

    def __getitem__(self, idx):
        from qwen_vl_utils import process_vision_info

        example = self.data[idx]
        messages = self._build_messages(example)

        # Full conversation text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Extract image tensors
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=False,
        )

        input_ids = inputs.input_ids[0]
        labels = input_ids.clone()

        # Mask everything up to (and including) the assistant marker so we only
        # supervise the assistant's response tokens.
        assistant_marker = "<|im_start|>assistant\n"
        marker_ids = self.processor.tokenizer.encode(
            assistant_marker, add_special_tokens=False
        )
        id_list = input_ids.tolist()
        marker_len = len(marker_ids)
        response_start = 0
        for i in range(len(id_list) - marker_len, -1, -1):
            if id_list[i : i + marker_len] == marker_ids:
                response_start = i + marker_len
                break
        labels[:response_start] = -100

        # Truncate if needed
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            attention_mask = inputs.attention_mask[0][: self.max_length]
        else:
            attention_mask = inputs.attention_mask[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": inputs.pixel_values,       # [N_patches, C*patch²]
            "image_grid_thw": inputs.image_grid_thw,   # [N_images, 3]
            "labels": labels,
        }


@dataclass
class MultiImageCollator:
    """Pads text tokens and concatenates image patch tensors for Qwen2-VL batching."""
    pad_token_id: int

    def __call__(self, examples):
        max_len = max(ex["input_ids"].shape[0] for ex in examples)

        batch = {k: [] for k in ("input_ids", "attention_mask", "labels",
                                  "pixel_values", "image_grid_thw")}

        for ex in examples:
            seq_len = ex["input_ids"].shape[0]
            pad_len = max_len - seq_len

            batch["input_ids"].append(
                torch.cat([ex["input_ids"], ex["input_ids"].new_full((pad_len,), self.pad_token_id)])
            )
            batch["attention_mask"].append(
                torch.cat([ex["attention_mask"], ex["attention_mask"].new_zeros(pad_len)])
            )
            batch["labels"].append(
                torch.cat([ex["labels"], ex["labels"].new_full((pad_len,), -100)])
            )
            # Qwen2-VL batches image patches by concatenation, not stacking
            batch["pixel_values"].append(ex["pixel_values"])
            batch["image_grid_thw"].append(ex["image_grid_thw"])

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "pixel_values": torch.cat(batch["pixel_values"], dim=0),
            "image_grid_thw": torch.cat(batch["image_grid_thw"], dim=0),
        }


def main():
    ModelClass = _import_model_class()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {MODEL_NAME} in 4-bit...")
    model = ModelClass.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        # Target the language model attention and MLP layers only (not the vision encoder)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_NAME, max_pixels=MAX_PIXELS)

    train_dataset = AIDetectionDataset(TRAIN_DATA, processor, MAX_SEQ_LEN)
    val_dataset = (
        AIDetectionDataset(VAL_DATA, processor, MAX_SEQ_LEN)
        if Path(VAL_DATA).exists()
        else None
    )
    print(f"Train: {len(train_dataset)} samples" +
          (f"  Val: {len(val_dataset)} samples" if val_dataset else ""))

    collator = MultiImageCollator(pad_token_id=processor.tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        dataloader_num_workers=0,      # required on Windows
        remove_unused_columns=False,   # keep pixel_values and image_grid_thw
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",        # 8-bit Adam saves ~0.5 GB vs fp32 Adam
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
