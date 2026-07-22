#!/usr/bin/env python3
"""Fine-tune BiRefNet on Sticker it cutline pairs (die-cut masks incl. bleed).

Runs on a RunPod GPU pod using the standard si-gpu-masking image (torch 2.6 + pinned
transformers + BiRefNet weights already cached — zero setup). The dataset arrives as
dataset.zip (from scripts/cutline-extract/make-dataset.py) with train/ and val/ splits.

  python3 train.py <dataset-dir> <out-dir> [epochs=40] [batch=4] [lr=3e-5]

- 1024x1024 (BiRefNet's native size), h-flip augmentation
- BCE-with-logits on ALL side outputs + soft-IoU on the final output
- fp16 autocast (matches production inference dtype), AdamW, cosine schedule
- saves best-by-val-IoU via save_pretrained → <out-dir>/best/  (ship these weights
  PRIVATELY — trained on customer data, never into the public image)
"""
import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

DATA = sys.argv[1]
OUT = sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 4
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-5
SIZE = 1024
DEV = "cuda"

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Pairs(Dataset):
    def __init__(self, split, augment):
        self.images = sorted(glob.glob(os.path.join(DATA, split, "images", "*.png")))
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        ip = self.images[i]
        mp = ip.replace(f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}")
        img = Image.open(ip).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        mask = Image.open(mp).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        x = norm(transforms.functional.to_tensor(img))
        y = torch.from_numpy((np.array(mask, dtype=np.float32) / 255.0)[None])
        return x, y


def soft_iou_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    union = (p + target - p * target).sum((2, 3))
    return (1 - (inter + 1) / (union + 1)).mean()


def out_list(out):
    return list(out) if isinstance(out, (list, tuple)) else [out]


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ious = []
    for x, y in loader:
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast(DEV, dtype=torch.float16):
            pred = out_list(model(x))[-1].float()
        m = (torch.sigmoid(pred) > 0.5).float()
        inter = (m * y).sum((2, 3))
        union = ((m + y) > 0).float().sum((2, 3))
        ious += (inter / union.clamp(min=1)).flatten().tolist()
    model.train()
    return sum(ious) / max(1, len(ious))


def main():
    os.makedirs(OUT, exist_ok=True)
    train = DataLoader(Pairs("train", True), batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val = DataLoader(Pairs("val", False), batch_size=BATCH, num_workers=2)
    print(f"train {len(train.dataset)} / val {len(val.dataset)} pairs, {EPOCHS} epochs, batch {BATCH}, lr {LR}")

    model = AutoModelForImageSegmentation.from_pretrained(
        os.environ.get("BG_HF_MODEL", "ZhengPeng7/BiRefNet"), trust_remote_code=True).to(DEV)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * len(train))
    scaler = torch.amp.GradScaler(DEV)

    base_iou = evaluate(model, val)
    print(f"BASELINE (stock BiRefNet) val IoU: {base_iou:.4f}")
    best = base_iou

    step = 0
    for ep in range(EPOCHS):
        t0, tot = time.time(), 0.0
        for x, y in train:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(DEV, dtype=torch.float16):
                outs = out_list(model(x))
                loss = sum(F.binary_cross_entropy_with_logits(o.float(), y) for o in outs) / len(outs)
                loss = loss + soft_iou_loss(outs[-1].float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item(); step += 1
        iou = evaluate(model, val)
        flag = ""
        if iou > best:
            best = iou
            model.save_pretrained(os.path.join(OUT, "best"))
            flag = "  ← saved"
        print(f"epoch {ep + 1}/{EPOCHS}  loss {tot / len(train):.4f}  val IoU {iou:.4f}  ({time.time() - t0:.0f}s){flag}", flush=True)

    print(f"\nDONE. baseline {base_iou:.4f} → best {best:.4f}  (weights in {OUT}/best)")


if __name__ == "__main__":
    main()
