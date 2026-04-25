"""
Image Style Transfer and Block Swap Pipeline.

Usage:
    python pipeline.py <input_image> [options]

This pipeline:
1. Takes an input image
2. Uses generative AI (Stable Diffusion + ControlNet) for style transfer
3. Feeds original and stylized images to block_swap for blending
"""

import argparse
import os
import random
import sys
import tempfile
from PIL import Image

from style_transfer import style_transfer
from block_swap import block_swap

D_OUTPUT_DIT = "output"
D_STRENGTH = 0.5
D_GUIDANCE = 7
D_PRESERVE_COLORS = False

D_GRID_SIZE = 256
D_SWAP_COUNT = 16384
D_CLUSTERED = True
D_NUM_CLUSTERS = 2

D_SEED = None

def run_pipeline(
    input_image: str,
    output_dir: str = None,
    images_dir: str = None,
    masks_dir: str = None,
    prompt: str = None,
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly",
    stylized_output: str = None,

    # Style transfer params
    use_controlnet: bool = True,
    preserve_colors: bool = D_PRESERVE_COLORS,
    strength: float = D_STRENGTH,
    steps: int = 30,
    guidance: float = D_GUIDANCE,
    device: str = None,

    # Block swap params
    grid_size: int = D_GRID_SIZE,
    swap_count: int = D_SWAP_COUNT,
    clustered: bool = D_CLUSTERED,
    num_clusters: int = D_NUM_CLUSTERS,
    seed: int = D_SEED,
) -> dict:
    """
    Run the full pipeline: style transfer + block swap.

    Args:
        input_image: Path to input image.
        output_dir: Directory for output files.
        prompt: Style transfer prompt.
        negative_prompt: Negative prompt for style transfer.
        stylized_output: Optional path to save/load stylized image.
        use_controlnet: Enable ControlNet for structure preservation.
        strength: Style transfer strength.
        steps: Style transfer inference steps.
        guidance: CFG guidance scale.
        device: Device to run on.
        grid_size: N for NxN grid in block swap.
        swap_count: Number of blocks to swap.
        clustered: Enable clustered block selection.
        num_clusters: Number of clusters.
        seed: Random seed for reproducibility.

    Returns:
        Dict with paths to generated files.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"Seed: {seed}")

    if output_dir is None:
        output_dir = os.path.dirname(input_image) or "."

    images_dir = images_dir or os.path.join(output_dir, "images")
    masks_dir = masks_dir or os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_image))[0]

    if stylized_output is None:
        stylized_output = os.path.join(output_dir, f"{base_name}_stylized.png")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input: {input_image}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Step 1: Style Transfer
    print("\n[Step 1/2] Running style transfer...")
    original_image, generated_image = style_transfer(
        input_image_path=input_image,
        output_path=stylized_output,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        device=device,
        use_controlnet=use_controlnet,
        preserve_colors=preserve_colors,
    )

    original_resized_path = os.path.join(output_dir, f"{base_name}_original_resized.png")
    original_image.save(original_resized_path)
    print(f"  - Stylized: {stylized_output}")
    print("-" * 50)

    # Step 2: Block Swap
    print("\n[Step 2/2] Running block swap...")
    import numpy as np
    block_swap_prefix = os.path.join(output_dir, f"{base_name}_swapped")
    arr1, arr2, mask = block_swap(
        img_path1=original_resized_path,
        img_path2=stylized_output,
        n=grid_size,
        m=swap_count,
        seed=seed,
        out_prefix=block_swap_prefix,
        clustered=clustered,
        num_clusters=num_clusters,
    )

    swapped1_path = f"{block_swap_prefix}_swapped1.png"
    swapped2_path = f"{block_swap_prefix}_swapped2.png"
    mask_png_path = f"{block_swap_prefix}_mask.png"

    images_copy_path = os.path.join(images_dir, f"{base_name}.png")
    mask_npy_path = os.path.join(masks_dir, f"{base_name}.npy")

    from PIL import Image as PILImage
    PILImage.fromarray(arr1).save(images_copy_path)
    np.save(mask_npy_path, (mask == 255).astype(np.uint8))
    print(f"  - images/: {images_copy_path}")
    print(f"  - masks/:  {mask_npy_path}")
    print("-" * 50)

    return {
        "original": original_resized_path,
        "stylized": stylized_output,
        "swapped1": swapped1_path,
        "swapped2": swapped2_path,
        "mask": mask_png_path,
        "images_copy": images_copy_path,
        "mask_npy": mask_npy_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Style transfer + block swap pipeline."
    )
    parser.add_argument("input", help="Path to input image or directory of images")
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory for swapped images (default: <output-dir>/images)"
    )
    parser.add_argument(
        "--masks-dir",
        default=None,
        help="Directory for .npy masks (default: <output-dir>/masks)"
    )

    # Style transfer options
    style_group = parser.add_argument_group("Style Transfer Options")
    style_group.add_argument(
        "-p", "--prompt",
        help="Text prompt for target style"
    )
    style_group.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted, deformed, ugly",
        help="Negative prompt"
    )
    style_group.add_argument(
        "--stylized-output",
        help="Path for stylized image (skip generation if exists)"
    )
    style_group.add_argument(
        "--no-controlnet",
        action="store_true",
        help="Disable ControlNet"
    )
    style_group.add_argument(
        "--no-preserve-colors",
        action="store_true",
        help="Disable color preservation (allow style to change colors)"
    )
    style_group.add_argument(
        "--strength",
        type=float,
        default=D_STRENGTH,
        help="Style transfer strength (0.0-1.0)"
    )
    style_group.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Inference steps"
    )
    style_group.add_argument(
        "--guidance",
        type=float,
        default=D_GUIDANCE,
        help="CFG guidance scale"
    )
    style_group.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device (default: auto)"
    )

    # Block swap options
    swap_group = parser.add_argument_group("Block Swap Options")
    swap_group.add_argument(
        "-n", "--grid-size",
        type=int,
        default=D_GRID_SIZE,
        help="Grid size N (NxN blocks)"
    )
    swap_group.add_argument(
        "-m", "--swap-count",
        type=int,
        default=D_SWAP_COUNT,
        help="Number of blocks to swap"
    )
    swap_group.add_argument(
        "-c", "--clusters",
        type=int,
        default=D_NUM_CLUSTERS,
        help="Number of clusters (enables clustered mode)"
    )
    swap_group.add_argument(
        "-s", "--seed",
        type=int,
        default=D_SEED,
        help="Random seed"
    )

    args = parser.parse_args()

    # Default prompt based on common use cases
    prompt = args.prompt
    if prompt is None:
        prompt = "anime style, detailed, vibrant colors, professional illustration, masterpiece"

    pipeline_kwargs = dict(
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        stylized_output=args.stylized_output,
        use_controlnet=not args.no_controlnet,
        preserve_colors=not args.no_preserve_colors,
        strength=args.strength,
        steps=args.steps,
        guidance=args.guidance,
        device=args.device,
        grid_size=args.grid_size,
        swap_count=args.swap_count,
        clustered=args.clusters is not None,
        num_clusters=args.clusters,
        seed=args.seed,
    )

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if os.path.isdir(args.input):
        inputs = sorted(
            p for p in (os.path.join(args.input, f) for f in os.listdir(args.input))
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTS
        )
        if not inputs:
            print(f"No images found in {args.input}")
            sys.exit(1)
        print(f"Found {len(inputs)} images in {args.input}\n")
        for i, img_path in enumerate(inputs, 1):
            print(f"[{i}/{len(inputs)}] Processing {img_path}")
            results = run_pipeline(input_image=img_path, **pipeline_kwargs)
            print("Files:")
            for name, path in results.items():
                print(f"  {name}: {path}")
            print()
    else:
        results = run_pipeline(input_image=args.input, **pipeline_kwargs)
        print("\n" + "=" * 50)
        print("Pipeline complete! Generated files:")
        for name, path in results.items():
            print(f"  {name}: {path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
