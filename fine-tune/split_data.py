"""
Split labeled.json into train.json and val.json (90/10 split by default).

Usage:
    python split_data.py --input data/labeled.json
    python split_data.py --input data/labeled.json --val-ratio 0.15 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--seed",      default=42, type=int)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    random.shuffle(data)

    n_val = max(1, int(len(data) * args.val_ratio))
    val   = data[:n_val]
    train = data[n_val:]

    out_dir = Path(args.input).parent
    for name, split in [("train", train), ("val", val)]:
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"{name}.json: {len(split)} samples → {path}")


if __name__ == "__main__":
    main()
