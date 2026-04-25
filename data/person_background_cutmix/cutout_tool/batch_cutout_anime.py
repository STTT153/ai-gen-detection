#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
批量为二次元肖像图去背景，导出 512x512 的透明 PNG。

默认会分别处理：
- C:\Users\86156\Desktop\new_inpaint\real
- C:\Users\86156\Desktop\new_inpaint\fake

并导出到：
- C:\Users\86156\Desktop\new_inpaint\cutout_test\real_png
- C:\Users\86156\Desktop\new_inpaint\cutout_test\fake_png

首次运行会自动下载 rembg 模型（默认 isnet-anime）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageOps
from rembg import new_session, remove

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Job:
    name: str
    input_dir: Path
    output_dir: Path


def collect_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def ensure_rgba_512(img: Image.Image) -> Image.Image:
    """
    保证输出为 RGBA 且尺寸为 512x512。
    如果原图不是 512x512，则用 contain 缩放后居中贴到透明画布。
    """
    img = ImageOps.exif_transpose(img).convert("RGBA")
    if img.size == (512, 512):
        return img

    contained = ImageOps.contain(img, (512, 512), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    x = (512 - contained.width) // 2
    y = (512 - contained.height) // 2
    canvas.alpha_composite(contained, (x, y))
    return canvas


def alpha_bbox(img: Image.Image, alpha_threshold: int = 8):
    alpha = img.getchannel("A")
    mask = alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
    return mask.getbbox()


def trim_and_refit(img: Image.Image, out_size=(512, 512), margin: int = 8) -> Image.Image:
    """
    将人物主体裁紧后重新放回 512x512 透明画布。
    只有在 --trim-subject 打开时使用。
    """
    bbox = alpha_bbox(img)
    if not bbox:
        return ensure_rgba_512(img)

    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)

    cropped = img.crop((left, top, right, bottom))
    contained = ImageOps.contain(cropped, out_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", out_size, (0, 0, 0, 0))
    x = (out_size[0] - contained.width) // 2
    y = (out_size[1] - contained.height) // 2
    canvas.alpha_composite(contained, (x, y))
    return canvas


def process_one(
    image_path: Path,
    output_path: Path,
    session,
    alpha_matting: bool,
    post_process_mask: bool,
    trim_subject: bool,
) -> None:
    with Image.open(image_path) as im:
        im = ensure_rgba_512(im)
        result = remove(
            im,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=post_process_mask,
        )

        if not isinstance(result, Image.Image):
            raise RuntimeError(f"处理失败，返回值不是 PIL.Image: {image_path}")

        result = result.convert("RGBA")
        result = trim_and_refit(result) if trim_subject else ensure_rgba_512(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, format="PNG")


def iter_limited(files: List[Path], limit: int | None) -> Iterable[Path]:
    if limit is None or limit <= 0:
        yield from files
    else:
        yield from files[:limit]


def run_job(
    job: Job,
    session,
    limit: int | None,
    alpha_matting: bool,
    post_process_mask: bool,
    trim_subject: bool,
) -> tuple[int, int]:
    if not job.input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {job.input_dir}")

    files = collect_images(job.input_dir)
    if not files:
        print(f"[警告] {job.name} 目录里没有找到可处理图片：{job.input_dir}")
        return 0, 0

    selected = list(iter_limited(files, limit))
    ok = 0
    fail = 0

    print(f"\n开始处理 [{job.name}]：共找到 {len(files)} 张，本次处理 {len(selected)} 张")
    print(f"输入目录：{job.input_dir}")
    print(f"输出目录：{job.output_dir}")

    for idx, image_path in enumerate(selected, start=1):
        out_name = image_path.stem + ".png"
        output_path = job.output_dir / out_name
        try:
            process_one(
                image_path=image_path,
                output_path=output_path,
                session=session,
                alpha_matting=alpha_matting,
                post_process_mask=post_process_mask,
                trim_subject=trim_subject,
            )
            ok += 1
            print(f"[{job.name}] {idx}/{len(selected)} OK  -> {output_path.name}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[{job.name}] {idx}/{len(selected)} FAIL -> {image_path.name} | {e}")

    return ok, fail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量扣出二次元人物并导出透明 PNG")
    parser.add_argument(
        "--real-dir",
        default=r"C:\Users\86156\Desktop\new_inpaint\real",
        help="真图目录",
    )
    parser.add_argument(
        "--fake-dir",
        default=r"C:\Users\86156\Desktop\new_inpaint\fake",
        help="假图目录",
    )
    parser.add_argument(
        "--output-root",
        default=r"C:\Users\86156\Desktop\new_inpaint\cutout_test",
        help="输出根目录，脚本会在下面创建 real_png 和 fake_png",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=60,
        help="每个类别处理多少张；设为 0 或负数表示全处理",
    )
    parser.add_argument(
        "--model",
        default="isnet-anime",
        help="rembg 模型名，二次元建议 isnet-anime",
    )
    parser.add_argument(
        "--no-alpha-matting",
        action="store_true",
        help="关闭 alpha matting（更快，但边缘可能稍差）",
    )
    parser.add_argument(
        "--no-post-process-mask",
        action="store_true",
        help="关闭 mask 后处理",
    )
    parser.add_argument(
        "--trim-subject",
        action="store_true",
        help="把人物主体裁紧后重新居中放回 512x512 透明画布",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    output_root = Path(args.output_root)

    alpha_matting = not args.no_alpha_matting
    post_process_mask = not args.no_post_process_mask
    limit = None if args.limit_per_class <= 0 else args.limit_per_class

    print("=" * 70)
    print("二次元人物透明背景批量导出")
    print(f"模型: {args.model}")
    print(f"alpha_matting: {alpha_matting}")
    print(f"post_process_mask: {post_process_mask}")
    print(f"trim_subject: {args.trim_subject}")
    print(f"limit_per_class: {args.limit_per_class}")
    print("=" * 70)
    print("提示：首次运行会下载模型，可能会稍慢。")

    try:
        session = new_session(args.model)
    except Exception as e:  # noqa: BLE001
        print(f"模型初始化失败: {e}")
        print("你可以先确认依赖安装是否成功，并检查网络是否能下载 rembg 模型。")
        return 2

    jobs = [
        Job("real", real_dir, output_root / "real_png"),
        Job("fake", fake_dir, output_root / "fake_png"),
    ]

    total_ok = 0
    total_fail = 0
    for job in jobs:
        ok, fail = run_job(
            job=job,
            session=session,
            limit=limit,
            alpha_matting=alpha_matting,
            post_process_mask=post_process_mask,
            trim_subject=args.trim_subject,
        )
        total_ok += ok
        total_fail += fail

    print("\n" + "=" * 70)
    print(f"完成：成功 {total_ok} 张，失败 {total_fail} 张")
    print(f"输出根目录：{output_root}")
    print("其中：")
    print(f"  真图输出：{output_root / 'real_png'}")
    print(f"  假图输出：{output_root / 'fake_png'}")
    print("=" * 70)

    return 0 if total_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
