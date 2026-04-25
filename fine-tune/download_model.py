"""
Download Qwen2.5-VL-7B-Instruct to ./models/Qwen2.5-VL-7B-Instruct

Two download sources (tries HuggingFace mirror first, falls back to ModelScope):
  - hf-mirror.com  (HuggingFace 镜像)
  - ModelScope     (魔搭，国内最稳)

Resume-safe: re-running continues from where it left off.

Usage:
    python download_model.py               # auto: try hf-mirror → modelscope
    python download_model.py --source hf   # force HuggingFace mirror
    python download_model.py --source ms   # force ModelScope
"""

import argparse
import os
import sys
from pathlib import Path

LOCAL_DIR  = str(Path(__file__).parent / "models" / "Qwen2.5-VL-7B-Instruct")
HF_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
MS_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"   # same ID on ModelScope


def download_hf():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import snapshot_download

    print("Source: hf-mirror.com")
    snapshot_download(
        repo_id=HF_MODEL,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.pt", "optimizer*", "training_args*"],
    )


def download_ms():
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print("Installing modelscope...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        from modelscope import snapshot_download as ms_download

    print("Source: ModelScope (魔搭)")
    ms_download(
        model_id=MS_MODEL,
        local_dir=LOCAL_DIR,
        ignore_file_pattern=["*.pt", "optimizer*", "training_args*"],
    )


def check_complete() -> bool:
    """Return True if weight shards are already present."""
    index = Path(LOCAL_DIR) / "model.safetensors.index.json"
    if not index.exists():
        return False
    import json
    with open(index) as f:
        weight_map = json.load(f).get("weight_map", {})
    shards = set(weight_map.values())
    return all((Path(LOCAL_DIR) / s).exists() for s in shards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "ms", "auto"], default="auto")
    args = parser.parse_args()

    print(f"Model : {HF_MODEL}")
    print(f"Target: {LOCAL_DIR}\n")

    if check_complete():
        print("Model weights already fully downloaded.")
        return

    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        download_hf()
    elif args.source == "ms":
        download_ms()
    else:
        # Auto: try HF mirror first, fall back to ModelScope
        try:
            download_hf()
            if not check_complete():
                raise RuntimeError("HF download incomplete")
        except Exception as e:
            print(f"\nhf-mirror failed ({e}), switching to ModelScope...\n")
            download_ms()

    if check_complete():
        print(f"\nDownload complete: {LOCAL_DIR}")
    else:
        print("\nWarning: some shards may be missing, re-run to resume.")


if __name__ == "__main__":
    main()
