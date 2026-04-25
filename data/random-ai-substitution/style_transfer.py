"""
Style Transfer using Stable Diffusion with ControlNet.

Preserves composition and subject position while changing art style.
Uses ControlNet (Canny/Depth) to maintain structural integrity.
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)


def transfer_color(source: Image.Image, target: Image.Image) -> Image.Image:
    """Replace target's colors with source's colors via LAB histogram matching."""
    import cv2
    src = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2LAB).astype(np.float32)
    # Match mean and std of A and B channels (color) from source onto target
    for ch in (1, 2):
        tgt[:, :, ch] = (tgt[:, :, ch] - tgt[:, :, ch].mean()) / (tgt[:, :, ch].std() + 1e-6) \
                        * src[:, :, ch].std() + src[:, :, ch].mean()
    tgt = np.clip(tgt, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(tgt, cv2.COLOR_LAB2RGB))


def load_pipeline(device: str = None):
    """Load Stable Diffusion + ControlNet pipeline."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use Canny ControlNet for edge-preserving style transfer
    local_controlnet_path = os.path.join(os.path.dirname(__file__), "models", "controlnet-canny")
    controlnet = ControlNetModel.from_pretrained(
        local_controlnet_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    local_model_path = os.path.join(os.path.dirname(__file__), "models", "sd-v1-5")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        local_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        # Enable attention slicing for memory efficiency
        pipe.enable_attention_slicing()

    return pipe, device


def detect_edges(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Detect edges in image using Canny algorithm."""
    import cv2
    import numpy as np

    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges).convert("RGB")


def style_transfer(
    input_image_path: str,
    output_path: str = None,
    prompt: str = None,
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly",
    strength: float = 0.5,
    num_inference_steps: int = 50,
    guidance_scale: float = 9,
    device: str = None,
    use_controlnet: bool = True,
    preserve_colors: bool = True,
) -> tuple[Image.Image, Image.Image]:
    """
    Perform style transfer on an input image.

    Args:
        input_image_path: Path to the input image.
        output_path: Optional path to save the output image.
        prompt: Text prompt describing the target style.
        negative_prompt: Negative prompt for quality control.
        strength: How much to transform the image (0.0-1.0).
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale for prompt adherence.
        device: Device to run on ('cuda' or 'cpu').
        use_controlnet: Whether to use ControlNet for structure preservation.

    Returns:
        Tuple of (original_image, generated_image).
    """
    if output_path is None:
        base, ext = os.path.splitext(input_image_path)
        output_path = f"{base}_stylized{ext}"

    if prompt is None:
        # Default prompt for artistic style transfer
        prompt = "anime style, detailed, vibrant colors, professional illustration, masterpiece"

    # Load input image
    original_image = Image.open(input_image_path).convert("RGB")

    # Ensure square dimensions for best results
    size = 512
    original_image = original_image.resize((size, size), Image.LANCZOS)

    if use_controlnet:
        # Load pipeline with ControlNet
        pipe, device = load_pipeline(device)

        # Generate edge map for ControlNet
        edge_image = detect_edges(original_image)

        # Generate stylized image
        generated_image = pipe(
            prompt=prompt,
            image=original_image,       # img2img base: preserves color/content
            control_image=edge_image,   # ControlNet input: edge map for structure
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    else:
        # Fallback: simple img2img without ControlNet (less structure preservation)
        from diffusers import StableDiffusionImg2ImgPipeline

        local_model_path = os.path.join(os.path.dirname(__file__), "models", "sd-v1-5")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        if device == "cuda":
            pipe.enable_model_cpu_offload()

        generated_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

    if preserve_colors:
        generated_image = transfer_color(original_image, generated_image)

    # Save results
    generated_image.save(output_path)
    print(f"Saved stylized image to: {output_path}")

    return original_image, generated_image


def main():
    parser = argparse.ArgumentParser(
        description="Style transfer using Stable Diffusion with ControlNet."
    )
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("-o", "--output", help="Path to output image")
    parser.add_argument(
        "-p", "--prompt",
        default="anime style, detailed, vibrant colors, professional illustration, masterpiece",
        help="Text prompt describing target style"
    )
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, distorted, deformed, ugly",
        help="Negative prompt"
    )
    parser.add_argument(
        "--no-controlnet",
        action="store_true",
        help="Disable ControlNet (use simple img2img instead)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Transformation strength (0.0-1.0, higher = more stylized)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="CFG guidance scale"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to run on (default: auto-detect)"
    )
    parser.add_argument(
        "--no-preserve-colors",
        action="store_true",
        help="Disable color preservation (allow style to change colors)"
    )

    args = parser.parse_args()

    style_transfer(
        input_image_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        device=args.device,
        use_controlnet=not args.no_controlnet,
        preserve_colors=not args.no_preserve_colors,
    )


if __name__ == "__main__":
    main()
