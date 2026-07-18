"""Vast.ai serverless PyWorker for the Sticker it GPU-masking service.

Launched by the Vast serverless start script (PYWORKER_REPO convention): it clones this
repo, installs requirements.txt, then runs `python worker.py`. To remove any dependency
on the template knowing how to start OUR model server, this file starts server.py itself
as a subprocess (logging to MODEL_LOG_FILE), then hands over to the Vast Worker, which
tails that log for MODEL_SERVER_READY before benchmarking.

Routes exposed (proxied to server.py on 127.0.0.1:18000):
  /v1/remove  — BiRefNet cutout (benchmark handler)
  /v1/mask    — raw BiRefNet mask (cutline / vectoriser)
  /v1/refine  — SAM 2 point & click mask

Cost model: constant workload 100 per request (jobs are single images of similar cost),
so capacity is effectively requests/second — matches `cost: 100` in /route/ calls.
"""
import base64
import io
import os
import random
import subprocess
import sys

from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "18000"))
MODEL_LOG_FILE = os.environ.get("MODEL_LOG_FILE", "/var/log/model/server.log")

# ---------------- start the model server ----------------
os.makedirs(os.path.dirname(MODEL_LOG_FILE), exist_ok=True)
_log = open(MODEL_LOG_FILE, "w")  # fresh log per worker start (PyWorker tails current run)
subprocess.Popen(
    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")],
    stdout=_log, stderr=subprocess.STDOUT,
    env={**os.environ, "MODEL_SERVER_PORT": str(MODEL_SERVER_PORT)},
)


# ---------------- benchmark payloads ----------------
def _benchmark_payload() -> dict:
    """A representative ~1024px image generated in-process (no network dependency):
    random blobs on a flat background — the sticker-artwork shape class."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1024, 1024), (240, 240, 240))
    d = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 8)):
        x, y = random.randint(100, 800), random.randint(100, 800)
        r = random.randint(60, 220)
        d.ellipse([x, y, x + r, y + r],
                  fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "feather": 1.0}


CONST_WORKLOAD = 100.0

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    handlers=[
        # /v1/remove — also the benchmark handler
        HandlerConfig(
            route="/v1/remove",
            allow_parallel_requests=True,   # FastAPI queues on the GPU; bursts absorb here
            max_queue_time=90.0,
            workload_calculator=lambda payload: CONST_WORKLOAD,
            benchmark_config=BenchmarkConfig(generator=_benchmark_payload, runs=4, concurrency=2),
        ),
        HandlerConfig(
            route="/v1/mask",
            allow_parallel_requests=True,
            max_queue_time=90.0,
            workload_calculator=lambda payload: CONST_WORKLOAD,
        ),
        HandlerConfig(
            route="/v1/refine",
            allow_parallel_requests=True,
            max_queue_time=30.0,            # interactive clicks: fail fast rather than queue
            workload_calculator=lambda payload: CONST_WORKLOAD,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["MODEL_SERVER_READY"],
        on_error=[
            "Traceback (most recent call last):",
            "torch.cuda.OutOfMemoryError",
            "RuntimeError: CUDA",
        ],
        on_info=["loading BiRefNet", "loading SAM 2"],
    ),
)

Worker(worker_config).run()
