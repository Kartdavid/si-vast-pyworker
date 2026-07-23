"""Model server for Vast.ai serverless — BiRefNet (auto cutout/mask) + SAM 2 (point & click).

Runs as a plain FastAPI app on 127.0.0.1:18000 behind the Vast PyWorker (worker.py),
which proxies JSON requests from the Vast serverless router to this server.

Startup design (matters on Vast): the HTTP port opens IMMEDIATELY and prints
MODEL_SERVER_READY; the models (~4 GB download on a fresh machine) load in a background
thread. Requests arriving before the models finish simply wait on the load lock. This is
required because Vast's benchmark probes the server shortly after startup — if the port
only opened after model load (the old design), slow hosts failed the benchmark with
"Cannot connect" and the worker got destroyed in a loop.

Endpoints (all JSON in / JSON out — the PyWorker forwards JSON payloads):
  GET  /health                            → {ok, loaded, models, gpu}
  POST /v1/remove  {image_b64|image_url, feather}            → {ok, image_b64, ms}
  POST /v1/mask    {image_b64|image_url, variant, threshold} → {ok, mask_b64, ms}
  POST /v1/refine  {image_b64|image_url, points:[{x,y,label}]} → {ok, mask_b64, ms}

Engines:
  - BiRefNet (ZhengPeng7/BiRefNet) @1024 — same model as the Modal service. Runs bf16 on
    Ampere+ GPUs, falls back to fp16 on older cards (pre-Ampere has no bf16), fp32 on CPU.
  - SAM 2.1 (facebook/sam2.1-hiera-large via transformers Sam2Model), fp32 — replaces the
    Modal service's SAM v1; label 1 = keep (green), 0 = remove (red).
"""
import base64
import io
import os
import threading
import time
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

try:  # iPhone HEIC support (roadmap item — graceful if the wheel is missing)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass

BG_MODEL = os.environ.get("BG_HF_MODEL", "ZhengPeng7/BiRefNet")
SAM2_MODEL = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-large")
SIZE = int(os.environ.get("BG_SIZE", "1024") or 1024)  # BiRefNet is trained at 1024
MAX_BYTES = int(os.environ.get("BG_MAX_BYTES", str(64 * 1024 * 1024)))
# Optional fine-tuned BiRefNet (our cutline-trained weights). If the safetensors file at
# FT_WEIGHTS_PATH exists at startup, it's loaded as a SECOND model (the stock hub model
# stays loaded too) and callers can select it per-request with {"model": FT_MODEL_NAME}.
# The architecture is built from the same hub id as stock; only the weights differ — the
# proven load_state_dict(strict=False) pattern from eval-worst.py. If the file is missing,
# nothing changes: only the stock model exists and every request serves it exactly as before.
# Default under /workspace: that's the persistent location the git-pulled code already lives
# in, so weights placed there survive the normal restart-and-git-pull cycle just like the code.
FT_MODEL_NAME = os.environ.get("FT_MODEL_NAME", "cutline-v1")
FT_WEIGHTS_PATH = os.environ.get("FT_WEIGHTS_PATH", "/workspace/cutline-v1/model.safetensors")
# Optional API key (RunPod / any direct-exposure deployment). If set, /v1/* requests must
# send it as the X-Api-Key header. On Vast this stays unset — the PyWorker's signature
# system is the gatekeeper there.
API_KEY = os.environ.get("API_KEY", "")

# ---------------- lazy model state ----------------
_lock = threading.Lock()
_loaded = False
_load_error = None
_birefnet = _birefnet_ft = _tf = _sam2 = _sam2_processor = None
_ft_error = None  # non-fatal: fine-tune load failed but stock is fine → serve stock
_device = "cpu"
_dtype = None


def _load_models():
    """Heavy imports + model downloads. Runs once (background thread at startup;
    requests block on the lock until it finishes)."""
    global _loaded, _load_error, _birefnet, _birefnet_ft, _ft_error, _tf, _sam2, _sam2_processor, _device, _dtype
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        if _load_error:  # don't retry a poisoned load — worker gets recycled instead
            raise RuntimeError(f"model_load_failed: {_load_error}")
        try:
            import torch
            from torchvision import transforms
            from transformers import AutoModelForImageSegmentation, Sam2Model, Sam2Processor

            _device = "cuda" if torch.cuda.is_available() else "cpu"
            if _device == "cuda":
                torch.set_float32_matmul_precision("high")
                # fp16, NOT bf16: BiRefNet's deformable-conv op has no BFloat16 kernel
                # ("deformable_im2col not implemented for 'BFloat16'"). fp16 is the model
                # card's recommended mode and ran the Modal deployment in production.
                _dtype = torch.float16
            else:
                _dtype = torch.float32
            torch.set_grad_enabled(False)

            print(f"loading BiRefNet ({BG_MODEL}) dtype={_dtype} device={_device} …", flush=True)
            _birefnet = (
                AutoModelForImageSegmentation.from_pretrained(BG_MODEL, trust_remote_code=True, dtype=_dtype)
                .to(_device)
                .eval()
            )
            _tf = transforms.Compose([
                transforms.Resize((SIZE, SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # Optional fine-tune: build the SAME architecture from the hub, then load OUR
            # weights into it. Failure here is NON-fatal — we log it and keep serving stock,
            # so a bad/partial weights file can never take the whole worker down.
            if os.path.exists(FT_WEIGHTS_PATH):
                try:
                    from safetensors.torch import load_file

                    print(f"loading fine-tune '{FT_MODEL_NAME}' from {FT_WEIGHTS_PATH} …", flush=True)
                    ft = AutoModelForImageSegmentation.from_pretrained(BG_MODEL, trust_remote_code=True, dtype=_dtype)
                    sd = load_file(FT_WEIGHTS_PATH)
                    missing, unexpected = ft.load_state_dict(sd, strict=False)
                    _birefnet_ft = ft.to(_device).eval()
                    print(f"FINE_TUNE_LOADED name={FT_MODEL_NAME} tensors={len(sd)} "
                          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
                except Exception as fe:  # non-fatal
                    _ft_error = str(fe)
                    _birefnet_ft = None
                    print(f"FINE_TUNE_LOAD_FAILED: {fe} — serving stock only", flush=True)
            else:
                print(f"no fine-tune at {FT_WEIGHTS_PATH} — stock only", flush=True)

            print(f"loading SAM 2 ({SAM2_MODEL}) …", flush=True)
            _sam2 = Sam2Model.from_pretrained(SAM2_MODEL).to(_device).eval()
            _sam2_processor = Sam2Processor.from_pretrained(SAM2_MODEL)

            _loaded = True
            print("MODELS_LOADED", flush=True)
            # Signal readiness ONLY now: the PyWorker's benchmark starts on this marker,
            # and it must run against loaded models (its requests time out if they have to
            # sit through the multi-GB model download; Vast allows ~38 min to reach this
            # line, so late is safe — early is what kills workers).
            print("MODEL_SERVER_READY", flush=True)
        except Exception as e:  # load failures are fatal for this worker — flag loudly
            _load_error = str(e)
            traceback.print_exc()
            raise


api = FastAPI(title="Sticker it — GPU masking", version="1.3.0")


@api.middleware("http")
async def _auth(request, call_next):
    if API_KEY and request.url.path.startswith("/v1/"):
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"ok": False, "error": "invalid_api_key"})
    return await call_next(request)


@api.on_event("startup")
def _on_startup():
    # Port opens right away (so nothing ever gets "connection refused"); the models warm
    # in the background and MODEL_SERVER_READY is printed only when they're loaded.
    print("MODEL_SERVER_STARTING", flush=True)
    threading.Thread(target=_load_models, daemon=True).start()


# ---------------- helpers ----------------
def _load_image(payload: dict):
    from PIL import Image

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


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_model(name):
    """Map a request's `model` param to a loaded net. Anything falsy or "stock" → stock.
    The fine-tune name → fine-tune IF it loaded, else transparently fall back to stock
    (a request must never fail just because the fine-tune is unavailable). Returns
    (net, served_name) so callers can report which model actually ran."""
    if name and name == FT_MODEL_NAME and _birefnet_ft is not None:
        return _birefnet_ft, FT_MODEL_NAME
    return _birefnet, "stock"


def _birefnet_mask(orig, net=None):
    """BiRefNet soft mask ('L', 0-255) at the original image size. `net` selects which
    loaded model to run (defaults to stock)."""
    import torch
    from PIL import Image

    net = net or _birefnet
    x = _tf(orig).unsqueeze(0).to(_device, dtype=_dtype)
    with torch.no_grad():
        out = net(x)
    pred = out[-1] if isinstance(out, (list, tuple)) else out
    pred = pred.float().sigmoid().cpu()[0].squeeze().numpy()
    # BILINEAR, not LANCZOS: upscaling a soft 1024px alpha mask to full res — visually
    # identical after feathering, and several times faster on big originals.
    return Image.fromarray((pred * 255).astype("uint8"), mode="L").resize(orig.size, Image.BILINEAR)


def _guarded(fn):
    """Run an endpoint body; request-level failures return a 500 JSON WITHOUT printing a
    traceback (the PyWorker treats 'Traceback' log lines as fatal worker errors — reserve
    that for model-load crashes, not one bad request)."""
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:
        print(f"REQUEST_ERROR: {type(e).__name__}: {e}", flush=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


# ---------------- endpoints ----------------
@api.get("/health")
def health():
    return {"ok": True, "loaded": _loaded, "load_error": _load_error, "birefnet": BG_MODEL,
            "sam2": SAM2_MODEL, "size": SIZE, "device": _device, "dtype": str(_dtype),
            # fine-tune visibility: ft_loaded True means callers can request it by name.
            "ft_model": FT_MODEL_NAME, "ft_loaded": _birefnet_ft is not None, "ft_error": _ft_error}


@api.post("/v1/remove")
def v1_remove(payload: dict):
    """Optional `result_put_url` (presigned PUT): the pod uploads the PNG straight to
    storage and returns tiny JSON instead of megabytes of base64 — the platform worker
    uses this so image bytes never round-trip through it."""
    def run():
        from PIL import ImageFilter

        t0 = time.perf_counter()
        _load_models()
        orig = _load_image(payload)
        net, served = _resolve_model(payload.get("model"))
        mask = _birefnet_mask(orig, net)
        feather = float(payload.get("feather", 1.0) or 0)
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        cut = orig.convert("RGBA")
        cut.putalpha(mask)
        put_url = payload.get("result_put_url")
        if put_url:
            import requests

            buf = io.BytesIO()
            # compress_level=1: PNG encoding of a full-res RGBA cutout at PIL's default
            # level was the single biggest cost on large images (~seconds). Level 1 is
            # 3-5x faster; the file is bigger but it's a transient working artifact.
            cut.save(buf, format="PNG", compress_level=1)
            r = requests.put(put_url, data=buf.getvalue(),
                             headers={"Content-Type": "image/png"}, timeout=60)
            if not r.ok:
                raise RuntimeError(f"result_upload_failed: {r.status_code}")
            return {"ok": True, "stored": True, "model": served, "ms": round((time.perf_counter() - t0) * 1000)}
        return {"ok": True, "image_b64": _png_b64(cut), "model": served,
                "ms": round((time.perf_counter() - t0) * 1000)}

    return _guarded(run)


@api.post("/v1/mask")
def v1_mask(payload: dict):
    """Raw BiRefNet mask — for the cutline mask-stage + vectoriser cutout.
    variant: "soft" (default; anti-aliased alpha) or "binary" (solid shape for tracing)."""
    def run():
        t0 = time.perf_counter()
        _load_models()
        orig = _load_image(payload)
        net, served = _resolve_model(payload.get("model"))
        mask = _birefnet_mask(orig, net)
        if payload.get("variant", "soft") == "binary":
            thr = int(payload.get("threshold", 128))
            mask = mask.point(lambda p: 255 if p >= thr else 0)
        return {"ok": True, "mask_b64": _png_b64(mask), "width": orig.size[0], "height": orig.size[1],
                "model": served, "ms": round((time.perf_counter() - t0) * 1000)}

    return _guarded(run)


@api.post("/v1/refine")
def v1_refine(payload: dict):
    """SAM 2 point & click: points=[{x,y,label}] in original-image pixels; label 1 = keep
    (green), 0 = remove (red). Returns the best candidate as a binary mask (subject=255)."""
    def run():
        import torch
        from PIL import Image

        t0 = time.perf_counter()
        pts = payload.get("points")
        if not isinstance(pts, list) or not pts:
            raise HTTPException(status_code=400, detail="points must be a non-empty list")
        _load_models()
        orig = _load_image(payload)

        # transformers SAM 2 shape: [image, object, point, xy] / labels [image, object, point]
        input_points = [[[[float(p["x"]), float(p["y"])] for p in pts]]]
        input_labels = [[[int(p.get("label", 1)) for p in pts]]]
        inputs = _sam2_processor(images=orig, input_points=input_points, input_labels=input_labels,
                                 return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _sam2(**inputs)
        masks = _sam2_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = outputs.iou_scores.cpu()[0][0]           # [num_masks]
        cand = masks[0].numpy().astype("uint8")           # [num_masks, H, W]
        # SAM 2 returns candidates at ~3 granularities (sub-part / part / whole). Raw
        # argmax-by-score loves the tiniest coherent thing under the click (a single
        # letter in a word). For touch-up UX, if the top-scoring mask is tiny and a
        # near-as-confident candidate is much larger, prefer the larger region.
        total = cand.shape[1] * cand.shape[2]
        areas = [int(c.sum()) for c in cand]
        # Candidate policy for touch-up clicks (learned from live testing 21 Jul):
        #  - a click on a letter shouldn't select one glyph  → promote larger candidates
        #  - a click on the background shouldn't select the WHOLE design and wipe it
        #    → hard ceiling at 40% of the canvas; if every candidate is huge, take the
        #      smallest one instead (the user can click again — never nuke everything).
        MAX_SHARE = 0.40
        ok = [i for i in range(len(areas)) if areas[i] <= MAX_SHARE * total]
        if not ok:
            best = int(min(range(len(areas)), key=lambda i: areas[i]))
        else:
            best = max(ok, key=lambda i: float(scores[i]))
            if areas[best] < 0.02 * total:
                for i in ok:
                    if i != best and float(scores[i]) >= float(scores[best]) - 0.15 and areas[i] >= 3 * areas[best]:
                        if areas[i] > areas[best]:
                            best = i
        m = cand[best] * 255
        mask = Image.fromarray(m, mode="L")
        return {"ok": True, "mask_b64": _png_b64(mask), "score": float(scores[best]),
                "ms": round((time.perf_counter() - t0) * 1000)}

    return _guarded(run)


if __name__ == "__main__":
    import uvicorn

    # 0.0.0.0 so a directly-exposed deployment (RunPod pod) is reachable; on Vast the port
    # isn't published, so this is equally safe behind the PyWorker.
    uvicorn.run(api, host=os.environ.get("MODEL_SERVER_HOST", "0.0.0.0"),
                port=int(os.environ.get("MODEL_SERVER_PORT", "18000")), log_level="warning")
