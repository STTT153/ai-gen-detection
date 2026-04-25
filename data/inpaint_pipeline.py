import os
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import sam_model_registry, SamPredictor
import cv2

# ========== 配置区（根据需要修改）==========
INPUT_DIR      = r"D:\Development\Projects\gen-inpaint-data\dataset"
OUTPUT_DIR     = r"D:\Development\Projects\gen-inpaint-data\output"
SAM_CHECKPOINT = r"D:\Development\Projects\gen-inpaint-data\sam_vit_b_01ec64.pth"
INPAINT_MODEL  = "runwayml/stable-diffusion-inpainting"

TASK_PROMPTS = {
    "face": (
        "anime face, beautiful detailed eyes, glossy lips, "
        "smooth skin, soft shading, cel shading, anime style, "
        "masterpiece, best quality, ultra detailed, sharp focus, "
        "illustration, key visual"
    ),
}
NEGATIVE_PROMPT = (
    "ugly, blurry, low quality, deformed, bad anatomy, "
    "realistic, photo, 3d render, western cartoon, "
    "extra fingers, fused fingers, watermark, text"
)
SELECTED_TASKS = ["face"]  # None 表示每张图随机选2种
# ============================================

DIR_ORIGINAL = os.path.join(OUTPUT_DIR, "original")
DIR_MASK     = os.path.join(OUTPUT_DIR, "mask")
DIR_INPAINT  = os.path.join(OUTPUT_DIR, "inpaint")
for d in [DIR_ORIGINAL, DIR_MASK, DIR_INPAINT]:
    os.makedirs(d, exist_ok=True)

print("加载 inpaint 模型...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    INPAINT_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.enable_attention_slicing()

print("加载 SAM ViT-B...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to("cuda")
predictor = SamPredictor(sam)

_anime_face_cascade = cv2.CascadeClassifier(
    r"D:\Development\Projects\gen-inpaint-data\lbpcascade_animeface.xml"
)

def get_face_mask(image_np):
    """检测动漫人脸，在裁剪区域上跑 SAM，返回精准 face mask"""
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _anime_face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=1, minSize=(16, 16)
    )

    if len(faces) == 0:
        # 回退：上方中心区域
        mask = np.zeros((h, w), dtype=bool)
        mask[h//8 : h//2, w//4 : w*3//4] = True
        return mask

    # 取面积最大的脸，加少量 padding
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    pad = int(min(fw, fh) * 0.15)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + fw + pad)
    y2 = min(h, y + fh + pad)

    # 裁剪到人脸区域，在小图上跑 SAM
    crop = image_np[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    predictor.set_image(crop)

    # 多个正样本点覆盖整张脸：中心、额头、左颊、右颊、下巴、左眼区、右眼区
    cx, cy = cw // 2, ch // 2
    points = np.array([
        [cx,      cy],           # 鼻子/脸颊中心
        [cx,      ch // 4],      # 额头
        [cx,      ch * 3 // 4],  # 下巴
        [cw // 4, ch // 2],      # 左颊
        [cw * 3 // 4, ch // 2],  # 右颊
        [cw // 3, ch // 3],      # 左眼区
        [cw * 2 // 3, ch // 3],  # 右眼区
    ])
    labels = np.ones(len(points), dtype=int)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    # 选包含脸部中心且面积最大的 mask（覆盖整张脸）
    valid = [m for m in masks if m[cy, cx]]
    crop_mask = max(valid, key=lambda m: m.sum()) if valid else masks[np.argmax(scores)]

    # 小核闭运算填充细小间隙
    crop_mask_u8 = crop_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    crop_mask_u8 = cv2.morphologyEx(crop_mask_u8, cv2.MORPH_CLOSE, kernel)

    # 将 crop 坐标系的 mask 映射回原图
    full_mask = np.zeros((h, w), dtype=bool)
    full_mask[y1:y2, x1:x2] = crop_mask_u8 > 0
    return full_mask

def process_image(image_path, tasks):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((512, 512))
    image_np = np.array(image_resized)
    stem = Path(image_path).stem

    image_resized.save(os.path.join(DIR_ORIGINAL, f"{stem}.png"))

    for task in tasks:
        print(f"  → 任务: {task}")
        try:
            if task == "face":
                mask = get_face_mask(image_np)
            else:
                raise NotImplementedError(f"Task '{task}' requires SAM point prompts")

            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            mask_image.save(os.path.join(DIR_MASK, f"{stem}_{task}.png"))

            result = pipe(
                prompt=TASK_PROMPTS[task],
                negative_prompt=NEGATIVE_PROMPT,
                image=image_resized,
                mask_image=mask_image,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            result_path = os.path.join(DIR_INPAINT, f"{stem}_{task}.png")
            result.save(result_path)
            print(f"  [OK] saved: {result_path}")

        except Exception as e:
            print(f"  [FAIL] task {task}: {e}")

task_list = list(TASK_PROMPTS.keys())
image_files = (
    list(Path(INPUT_DIR).glob("*.png")) +
    list(Path(INPUT_DIR).glob("*.jpg")) +
    list(Path(INPUT_DIR).glob("*.jpeg"))
)

print(f"\n共找到 {len(image_files)} 张图片，开始处理...\n")

for i, image_path in enumerate(image_files):
    selected_tasks = SELECTED_TASKS if SELECTED_TASKS else random.sample(task_list, 2)
    stem = Path(image_path).stem
    if all(os.path.exists(os.path.join(DIR_INPAINT, f"{stem}_{t}.png")) for t in selected_tasks):
        print(f"[{i+1}/{len(image_files)}] {image_path.name} → 跳过（已完成）")
        continue
    print(f"[{i+1}/{len(image_files)}] {image_path.name} → 任务: {selected_tasks}")
    process_image(str(image_path), selected_tasks)
    torch.cuda.empty_cache()

print("\n全部完成！")
