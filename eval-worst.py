#!/usr/bin/env python3
"""Worst-case eval: per-image IoU of the fine-tuned model on the val set, worst offenders
rendered as side-by-side sheets (artwork | human mask | model mask). The tail of this
distribution — not the mean — is what the future instant-proofing confidence gate must
catch.

  python3 eval-worst.py <dataset-dir> <weights-dir> <out-dir>
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

DATA, WEIGHTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SIZE = 1024
DEV = "cuda"
os.makedirs(OUT, exist_ok=True)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Local save of a custom-code model loses its model_type — so build the ARCHITECTURE
# from the hub and load OUR fine-tuned weights into it (production loads them the same way).
from safetensors.torch import load_file

model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
sd = load_file(os.path.join(WEIGHTS, "model.safetensors"))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded fine-tuned weights: {len(sd)} tensors ({len(missing)} missing, {len(unexpected)} unexpected)")
model = model.to(DEV).eval()
torch.set_grad_enabled(False)

rows = []
for ip in sorted(glob.glob(os.path.join(DATA, "val", "images", "*.png"))):
    name = os.path.basename(ip)[:-4]
    mp = os.path.join(DATA, "val", "masks", name + ".png")
    img = Image.open(ip).convert("RGB")
    gt = Image.open(mp).convert("L")
    x = norm(transforms.functional.to_tensor(img.resize((SIZE, SIZE), Image.BILINEAR)))[None].to(DEV)
    with torch.autocast(DEV, dtype=torch.float16):
        out = model(x)
        o = (out[-1] if isinstance(out, (list, tuple)) else out).float()
    pred = torch.sigmoid(o)[0, 0].cpu().numpy()
    pred_img = Image.fromarray((pred * 255).astype("uint8")).resize(img.size, Image.BILINEAR)
    p = np.array(pred_img, dtype=np.float32) / 255.0 > 0.5
    g = np.array(gt, dtype=np.float32) / 255.0 > 0.5
    inter, union = (p & g).sum(), (p | g).sum()
    iou = inter / max(1, union)
    rows.append((iou, name, img, gt, pred_img))

rows.sort(key=lambda r: r[0])
ious = [r[0] for r in rows]
print(f"val n={len(ious)}  mean {np.mean(ious):.4f}  p50 {np.percentile(ious,50):.4f}  "
      f"p10 {np.percentile(ious,10):.4f}  p5 {np.percentile(ious,5):.4f}  min {min(ious):.4f}")
print("\nworst 10:")
for iou, name, *_ in rows[:10]:
    print(f"  {iou:.4f}  {name}")

for iou, name, img, gt, pred_img in rows[:8]:
    w, h = img.size
    sheet = Image.new("RGB", (w * 3 + 20, h + 30), (245, 245, 248))
    for i, im in enumerate([img, gt.convert("RGB"), pred_img.convert("RGB")]):
        sheet.paste(im, (i * (w + 10), 24))
    ImageDraw.Draw(sheet).text((6, 4), f"{name}  IoU {iou:.4f}  | artwork — human mask — model mask", fill=(20, 20, 40))
    sheet.save(os.path.join(OUT, f"{iou:.3f}-{name}.png"))
print(f"\nworst-8 sheets → {OUT}")
