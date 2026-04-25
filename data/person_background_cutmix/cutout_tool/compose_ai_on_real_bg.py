import os
import random
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# =========================
# 1. 路径设置
# =========================
AI_FG_DIR = r"C:\Users\86156\Desktop\new_inpaint\cutout_test\fake_png"
REAL_BG_DIR = r"C:\Users\86156\Desktop\new_inpaint\background\real_background"
OUTPUT_DIR = r"C:\Users\86156\Desktop\new_inpaint\output\ai_on_real_512"

OUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MASK_PNG_DIR = os.path.join(OUTPUT_DIR, "masks_png")   # 黑白png蒙版
OUT_MASK_NPY_DIR = os.path.join(OUTPUT_DIR, "masks_npy")   # 0/1矩阵
OUT_META_DIR = os.path.join(OUTPUT_DIR, "meta")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_PNG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_NPY_DIR, exist_ok=True)
os.makedirs(OUT_META_DIR, exist_ok=True)

# =========================
# 2. 参数设置
# =========================
TARGET_SIZE = (512, 512)
NUM_OUTPUTS = 300

# 人物填充模式：
# "contain" = 完整保留人物，不裁切，尽量接近四边
# "cover"   = 尽量铺满整个512x512，可能会裁掉一部分
FG_FIT_MODE = "cover"

# 四周预留边距
FG_MARGIN = 0

# 人物位置：
# "center" = 居中
# "bottom" = 稍微靠下，更像站立
FG_ALIGN = "center"

# 是否随机翻转
ENABLE_RANDOM_FLIP = True

# 是否添加阴影
ENABLE_SHADOW = False

# mask二值化阈值
# alpha > 127 记为前景(1)，否则背景(0)
MASK_THRESHOLD = 127

random.seed(42)


def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    for p in Path(folder).iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    return files


def resize_and_crop_bg(img, target_size=(512, 512)):
    """
    背景图缩放并中心裁剪到固定尺寸
    """
    tw, th = target_size
    w, h = img.size

    scale = max(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    right = left + tw
    bottom = top + th

    img = img.crop((left, top, right, bottom))
    return img


def trim_transparent_border(fg_rgba):
    """
    裁掉人物图外围透明区域
    """
    alpha = fg_rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return fg_rgba
    return fg_rgba.crop(bbox)


def make_shadow(fg_rgba, shadow_blur=10, alpha_scale=0.25):
    """
    根据人物 alpha 生成简单阴影
    """
    alpha = fg_rgba.getchannel("A")
    alpha_np = np.array(alpha).astype(np.float32)
    alpha_np = (alpha_np * alpha_scale).clip(0, 255).astype(np.uint8)

    shadow_alpha = Image.fromarray(alpha_np, mode="L")
    shadow = Image.new("RGBA", fg_rgba.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    return shadow


def alpha_bbox_from_binary(mask_binary):
    """
    根据0/1二值矩阵计算bbox
    返回格式: [xmin, ymin, xmax, ymax]
    """
    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def fit_fg_to_canvas(fg_rgba, target_size=(512, 512), mode="contain", margin=8):
    """
    把透明人物图缩放到目标画布内
    mode:
      - contain: 人物完整保留，不裁
      - cover:   人物铺满画布，可能裁边
    """
    tw, th = target_size
    fg_w, fg_h = fg_rgba.size

    avail_w = max(1, tw - 2 * margin)
    avail_h = max(1, th - 2 * margin)

    if mode == "contain":
        scale = min(avail_w / fg_w, avail_h / fg_h)
    elif mode == "cover":
        scale = max(avail_w / fg_w, avail_h / fg_h)
    else:
        raise ValueError("mode 必须是 'contain' 或 'cover'")

    new_w = max(1, int(fg_w * scale))
    new_h = max(1, int(fg_h * scale))
    fg_rgba = fg_rgba.resize((new_w, new_h), Image.LANCZOS)
    return fg_rgba


def paste_with_crop_rgba(base_rgba, overlay_rgba, x, y):
    """
    支持负坐标粘贴，超出部分自动裁掉
    """
    bw, bh = base_rgba.size
    ow, oh = overlay_rgba.size

    left = max(0, x)
    top = max(0, y)
    right = min(bw, x + ow)
    bottom = min(bh, y + oh)

    if right <= left or bottom <= top:
        return base_rgba

    crop_left = left - x
    crop_top = top - y
    crop_right = crop_left + (right - left)
    crop_bottom = crop_top + (bottom - top)

    overlay_crop = overlay_rgba.crop((crop_left, crop_top, crop_right, crop_bottom))
    base_rgba.alpha_composite(overlay_crop, (left, top))
    return base_rgba


def paste_with_crop_mask(base_mask, overlay_mask, x, y):
    """
    支持负坐标粘贴 mask
    """
    bw, bh = base_mask.size
    ow, oh = overlay_mask.size

    left = max(0, x)
    top = max(0, y)
    right = min(bw, x + ow)
    bottom = min(bh, y + oh)

    if right <= left or bottom <= top:
        return base_mask

    crop_left = left - x
    crop_top = top - y
    crop_right = crop_left + (right - left)
    crop_bottom = crop_top + (bottom - top)

    overlay_crop = overlay_mask.crop((crop_left, crop_top, crop_right, crop_bottom))
    base_mask.paste(overlay_crop, (left, top))
    return base_mask


def get_fg_position(bg_size, fg_size, align="bottom", margin=8):
    """
    计算人物在512画布中的放置位置
    """
    bw, bh = bg_size
    fw, fh = fg_size

    x = (bw - fw) // 2

    if align == "center":
        y = (bh - fh) // 2
    elif align == "bottom":
        y = bh - fh - margin
    else:
        raise ValueError("align 必须是 'center' 或 'bottom'")

    return x, y


def compose_one(bg_path, fg_path, out_idx):
    # 1) 背景
    bg = Image.open(bg_path).convert("RGB")
    bg = resize_and_crop_bg(bg, TARGET_SIZE)
    bg_w, bg_h = bg.size

    # 2) 前景
    fg = Image.open(fg_path).convert("RGBA")
    fg = trim_transparent_border(fg)

    if ENABLE_RANDOM_FLIP and random.random() < 0.5:
        fg = fg.transpose(Image.FLIP_LEFT_RIGHT)

    # 3) 人物缩放到512画布
    fg = fit_fg_to_canvas(
        fg_rgba=fg,
        target_size=TARGET_SIZE,
        mode=FG_FIT_MODE,
        margin=FG_MARGIN
    )

    fg_w, fg_h = fg.size

    # 4) 放置位置
    x, y = get_fg_position(
        bg_size=(bg_w, bg_h),
        fg_size=(fg_w, fg_h),
        align=FG_ALIGN,
        margin=FG_MARGIN
    )

    # 5) 合成图
    canvas = bg.convert("RGBA")

    if ENABLE_SHADOW:
        shadow = make_shadow(fg, shadow_blur=8, alpha_scale=0.22)
        shadow_x = x + 6
        shadow_y = y + 8
        canvas = paste_with_crop_rgba(canvas, shadow, shadow_x, shadow_y)

    canvas = paste_with_crop_rgba(canvas, fg, x, y)

    # 6) 生成原始灰度mask
    mask = Image.new("L", (bg_w, bg_h), 0)
    fg_alpha = fg.getchannel("A")
    mask = paste_with_crop_mask(mask, fg_alpha, x, y)

    # 7) 灰度mask -> 0/1 numpy矩阵
    mask_np = np.array(mask)
    mask_01 = (mask_np > MASK_THRESHOLD).astype(np.uint8)   # 值只有0和1

    # 8) 0/1矩阵 -> 黑白png (0/255)
    mask_png = Image.fromarray(mask_01 * 255, mode="L")

    # 9) bbox和面积
    bbox = alpha_bbox_from_binary(mask_01)
    area_ratio = float(mask_01.sum() / (bg_w * bg_h))

    # 10) 文件名
    image_name = f"{out_idx:06d}.png"
    mask_png_name = f"{out_idx:06d}_mask.png"
    mask_npy_name = f"{out_idx:06d}_mask.npy"
    meta_name = f"{out_idx:06d}.json"

    out_img_path = os.path.join(OUT_IMAGE_DIR, image_name)
    out_mask_png_path = os.path.join(OUT_MASK_PNG_DIR, mask_png_name)
    out_mask_npy_path = os.path.join(OUT_MASK_NPY_DIR, mask_npy_name)
    out_meta_path = os.path.join(OUT_META_DIR, meta_name)

    # 11) 保存
    canvas.convert("RGB").save(out_img_path)
    mask_png.save(out_mask_png_path)
    np.save(out_mask_npy_path, mask_01)

    # 12) meta
    meta = {
        "image_id": f"{out_idx:06d}",
        "image_path": out_img_path,
        "mask_png_path": out_mask_png_path,
        "mask_npy_path": out_mask_npy_path,
        "image_label": 1,
        "sample_type": "real_bg_ai_char",
        "bg_source": os.path.basename(bg_path),
        "fg_source": os.path.basename(fg_path),
        "fit_mode": FG_FIT_MODE,
        "align_mode": FG_ALIGN,
        "margin": FG_MARGIN,
        "mask_threshold": MASK_THRESHOLD,
        "position": {
            "x": x,
            "y": y
        },
        "bbox": bbox,
        "area_ratio": area_ratio,
        "target_size": {
            "width": bg_w,
            "height": bg_h
        }
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return out_img_path, out_mask_png_path, out_mask_npy_path


def main():
    print("脚本启动成功")

    bg_list = list_images(REAL_BG_DIR)
    fg_list = list_images(AI_FG_DIR)

    print("背景图数量：", len(bg_list))
    print("前景图数量：", len(fg_list))

    if len(bg_list) == 0:
        raise ValueError(f"背景文件夹里没有图片：{REAL_BG_DIR}")
    if len(fg_list) == 0:
        raise ValueError(f"前景文件夹里没有图片：{AI_FG_DIR}")

    print(f"准备生成 {NUM_OUTPUTS} 张 512x512 合成图...\n")

    for i in range(NUM_OUTPUTS):
        bg_path = random.choice(bg_list)
        fg_path = random.choice(fg_list)

        try:
            out_img, out_mask_png, out_mask_npy = compose_one(bg_path, fg_path, i)
            print(f"[{i+1}/{NUM_OUTPUTS}] 完成:")
            print(f"  图像: {out_img}")
            print(f"  PNG蒙版: {out_mask_png}")
            print(f"  NPY矩阵: {out_mask_npy}")
        except Exception as e:
            print(f"[{i+1}/{NUM_OUTPUTS}] 失败: {e}")

    print("\n全部完成")
    print("输出目录：", OUTPUT_DIR)
    print("图像目录：", OUT_IMAGE_DIR)
    print("PNG蒙版目录：", OUT_MASK_PNG_DIR)
    print("NPY矩阵目录：", OUT_MASK_NPY_DIR)
    print("Meta目录：", OUT_META_DIR)


if __name__ == "__main__":
    main()