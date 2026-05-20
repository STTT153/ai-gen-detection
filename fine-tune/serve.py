"""
OpenAI-compatible inference server for the fine-tuned Qwen2.5-VL model.

Exposes a /v1/chat/completions endpoint so web/app.py can call it via
the LOCAL_LLM_BASE_URL startup option.

Usage:
    cd fine-tune
    python serve.py                                   # base model only
    python serve.py --adapter output/qwen-vl-lora    # with LoRA adapter
    python serve.py --adapter output/qwen-vl-lora --port 8001 --host 0.0.0.0

Then start the web app pointing at it:
    LOCAL_LLM_BASE_URL=http://localhost:8001/v1 \\
    LOCAL_LLM_MODEL=qwen-finetuned \\
    LOCAL_LLM_API_KEY=local \\
    python web/app.py
"""

from __future__ import annotations

import argparse
import base64
import io
import time
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from inference import load_model

app = FastAPI(title="Qwen2.5-VL fine-tuned server")

_model = None
_processor = None


# ── OpenAI-compatible request/response schemas ───────────────────────────────

class _ImageURL(BaseModel):
    url: str  # "data:image/jpeg;base64,<data>"


class _ContentPart(BaseModel):
    type: str                          # "text" | "image_url"
    text: Optional[str] = None
    image_url: Optional[_ImageURL] = None


class _Message(BaseModel):
    role: str
    content: str | list[_ContentPart]


class ChatRequest(BaseModel):
    model: str = "qwen-finetuned"
    messages: list[_Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64url_to_pil(data_url: str):
    """Convert 'data:image/jpeg;base64,...' to PIL Image."""
    from PIL import Image
    _, encoded = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def _to_qwen_messages(messages: list[_Message]) -> list[dict]:
    """Convert OpenAI message list to Qwen2.5-VL message list.

    llm.py sends images as image_url with base64 data URLs.
    Qwen processor expects {"type": "image", "image": <PIL.Image>}.
    """
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            result.append({"role": msg.role, "content": msg.content})
            continue

        content = []
        for part in msg.content:
            if part.type == "text" and part.text:
                content.append({"type": "text", "text": part.text})
            elif part.type == "image_url" and part.image_url:
                pil_img = _b64url_to_pil(part.image_url.url)
                content.append({"type": "image", "image": pil_img})

        result.append({"role": msg.role, "content": content})
    return result


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "qwen-finetuned", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    from qwen_vl_utils import process_vision_info

    qwen_messages = _to_qwen_messages(req.messages)

    text = _processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(qwen_messages)

    device = next(_model.parameters()).device
    inputs = _processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=True,
            top_p=req.top_p,
        )

    generated = output_ids[:, inputs.input_ids.shape[1]:]
    text_out = _processor.batch_decode(generated, skip_special_tokens=True)[0]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text_out},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _model, _processor

    parser = argparse.ArgumentParser(description="Serve fine-tuned Qwen2.5-VL model")
    parser.add_argument(
        "--base-model",
        default="models/Qwen2.5-VL-7B-Instruct",
        help="Base model name or local path (default: models/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA adapter directory (default: none, use base model only)",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"Loading base model : {args.base_model}")
    if args.adapter:
        print(f"LoRA adapter       : {args.adapter}")
    else:
        print("LoRA adapter       : none (base model only)")

    _model, _processor = load_model(args.base_model, args.adapter)

    print(f"\nServer ready — http://{args.host}:{args.port}/v1")
    print("Connect web app with:")
    print(f"  LOCAL_LLM_BASE_URL=http://{args.host}:{args.port}/v1 \\")
    print( "  LOCAL_LLM_MODEL=qwen-finetuned \\")
    print( "  python web/app.py\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
