"""Model server for Vast.ai serverless — BiRefNet (auto cutout/mask) + SAM 2 (point & click).

Runs as a plain FastAPI app on 127.0.0.1:18000 behind the Vast PyWorker (worker.py),
which proxies JSON requests from the Vast serverless router to this server.

Endpoints (all JSON in / JSON out — the PyWorker forwards JSON payloads):
  GET  /health                            → {ok, models, gpu}
  POST /v1/remove  {image_b64|image_url, feather}          → {ok, image_b64, ms}
  POST /v1/mask    {image_b64|image_url, variant, threshold} → {ok, mask_b64, ms}
  POST /v1/refine  {image_b64|image_url, points:[{x,y,label}]} → {ok, mask_b64, ms}

Engines:
  - BiRefNet (ZhengPeng7/BiRefNet) in bf16 @1024 — same model as the Modal service,
    Byron's cleaner bf16 dtype (no fp16 half() workaround needed).
  - SAM 2.1 (facebook/sam2.1-hiera-large via transformers Sam2Model) — replaces the
    Modal service's SAM v1; label 1 = keep (green), 0 = remove (red).

Prints MODEL_SERVER_READY when both models are loaded — worker.py's LogActionConfig
watches for that line before benchmarking.
"""
import base64
import io
import os
import sys
import time

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageFilter

try:  # iPhone HEIC support (roadmap item — graceful if the wheel is missing)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BG_MODEL = os.environ.get("BG_HF_MODEL", "ZhengPeng7/BiRefNet")
SAM2_MODEL = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-large")
SIZE = int(os.environ.get("BG_SIZE", "1024") or 1024)  # BiRefNet is trained at 1024
MAX_BYTES = int(os.environ.get("BG_MAX_BYTES", str(64 * 1024 * 1024)))

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

print(f"loading BiRefNet ({BG_MODEL}) …", flush=True)
from torchvision import transforms  # noqa: E402
from transformers import AutoModelForImageSegmentation, Sam2Model, Sam2Processor  # noqa: E402

_birefnet = (
    AutoModelForImageSegmentation.from_pretrained(BG_MODEL, trust_remote_code=True, dtype=torch.bfloat16)
    .to(DEVICE)
    .eval()
)
_tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print(f"loading SAM 2 ({SAM2_MODEL}) …", flush=True)
_sam2 = Sam2Model.from_pretrained(SAM2_MODEL).to(DEVICE).eval()
_sam2_processor = Sam2Processor.from_pretrained(SAM2_MODEL)

print("MODEL_SERVER_READY", flush=True)
sys.stdout.flush()

api = FastAPI(title="Sticker it — GPU masking (Vast)", version="1.0.0")


# ---------------- helpers ----------------
def _load_image(payload: dict) -> Image.Image:
    b64 = payload.get("image_b64")
    url = payload.get("image_url")
    if b64:
        try:
            data = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="image_b64 is not valid base64")
    elif url:
        if not (url.startswith("http://") or url.startswith("https://")):
            raise HTTPException(status_code=400, detail="image_url must be http(s)")
        import requests

        r = requests.get(url, timeout=30)
        if not r.ok:
            raise HTTPException(status_code=400, detail=f"fetch_failed: {r.status_code}")
        data = r.content
    else:
        raise HTTPException(status_code=400, detail="provide image_b64 or image_url")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="file_too_large")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        # Same contract as the Modal service: undecodable bytes = user error (400),
        # which the platform worker maps to `unsupported_image`.
        raise HTTPException(status_code=400, detail="cannot identify image file")


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _birefnet_mask(orig: Image.Image) -> Image.Image:
    """BiRefNet soft mask ('L', 0-255) at the original image size."""
    x = _tf(orig).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    out = _birefnet(x)
    pred = out[-1] if isinstance(out, (list, tuple)) else out
    pred = pred.float().sigmoid().cpu()[0].squeeze().numpy()
    return Image.fromarray((pred * 255).astype("uint8"), mode="L").resize(orig.size, Image.LANCZOS)


# ---------------- endpoints ----------------
@api.get("/health")
def health():
    return {"ok": True, "birefnet": BG_MODEL, "sam2": SAM2_MODEL, "size": SIZE,
            "gpu": torch.cuda.is_available(), "engine": "pytorch-bf16"}


@api.post("/v1/remove")
def v1_remove(payload: dict):
    t0 = time.perf_counter()
    orig = _load_image(payload)
    mask = _birefnet_mask(orig)
    feather = float(payload.get("feather", 1.0) or 0)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    cut = orig.convert("RGBA")
    cut.putalpha(mask)
    return {"ok": True, "image_b64": _png_b64(cut), "ms": round((time.perf_counter() - t0) * 1000)}


@api.post("/v1/mask")
def v1_mask(payload: dict):
    """Raw BiRefNet mask — for the cutline mask-stage + vectoriser cutout.
    variant: "soft" (default; anti-aliased alpha) or "binary" (solid shape for tracing)."""
    t0 = time.perf_counter()
    orig = _load_image(payload)
    mask = _birefnet_mask(orig)
    if payload.get("variant", "soft") == "binary":
        thr = int(payload.get("threshold", 128))
        mask = mask.point(lambda p: 255 if p >= thr else 0)
    return {"ok": True, "mask_b64": _png_b64(mask), "width": orig.size[0], "height": orig.size[1],
            "ms": round((time.perf_counter() - t0) * 1000)}


@api.post("/v1/refine")
def v1_refine(payload: dict):
    """SAM 2 point & click: points=[{x,y,label}] in original-image pixels; label 1 = keep
    (green), 0 = remove (red). Returns the best candidate as a binary mask (subject=255)."""
    t0 = time.perf_counter()
    pts = payload.get("points")
    if not isinstance(pts, list) or not pts:
        raise HTTPException(status_code=400, detail="points must be a non-empty list")
    orig = _load_image(payload)

    # transformers SAM 2 shape: [image, object, point, xy] / labels [image, object, point]
    input_points = [[[[float(p["x"]), float(p["y"])] for p in pts]]]
    input_labels = [[[int(p.get("label", 1)) for p in pts]]]
    inputs = _sam2_processor(images=orig, input_points=input_points, input_labels=input_labels,
                             return_tensors="pt").to(DEVICE)
    outputs = _sam2(**inputs)
    masks = _sam2_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    scores = outputs.iou_scores.cpu()[0][0]           # [num_masks]
    best = int(scores.argmax())
    m = masks[0][best].numpy().astype("uint8") * 255  # [num_masks, H, W] → best
    mask = Image.fromarray(m, mode="L")
    return {"ok": True, "mask_b64": _png_b64(mask), "score": float(scores[best]),
            "ms": round((time.perf_counter() - t0) * 1000)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="127.0.0.1", port=int(os.environ.get("MODEL_SERVER_PORT", "18000")),
                log_level="warning")
