"""Smoke-test client for the Vast serverless GPU-masking endpoint.

Usage (from apps/bg-remove, with the key in .env.vast):
  python3 vast/test_client.py health
  python3 vast/test_client.py remove path/to/image.png          → vast-cutout.png
  python3 vast/test_client.py mask path/to/image.png            → vast-mask.png
  python3 vast/test_client.py refine path/to/image.png 350 350  → vast-refine-mask.png

Flow (what the platform worker will do in Node):
  1. POST https://run.vast.ai/route/ {endpoint, cost:100}  (Bearer VAST_API_KEY)
  2. POST {worker_url}{route} {auth_data, payload}
"""
import base64
import json
import os
import sys
import time
import urllib.request

ENDPOINT_NAME = os.environ.get("VAST_ENDPOINT", "si-gpu-masking-staging")
ROUTE_URL = "https://run.vast.ai/route/"


def _env_file(name: str) -> str | None:
    """Read KEY=value lines from .env.vast (next to this script or cwd) + real env vars."""
    if os.environ.get(name):
        return os.environ[name]
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, ".env.vast"), os.path.join(here, "..", ".env.vast"), ".env.vast"):
        if os.path.exists(p):
            for line in open(p):
                if line.strip().startswith(f"{name}="):
                    return line.strip().split("=", 1)[1]
    return None


# Direct mode (RunPod pod or any plain host): set GPU_MASK_URL (+ GPU_MASK_KEY) in
# .env.vast and the client POSTs straight to {url}/v1/... with the X-Api-Key header —
# no Vast routing involved.
DIRECT_URL = _env_file("GPU_MASK_URL")
DIRECT_KEY = _env_file("GPU_MASK_KEY")


def _api_key() -> str:
    key = _env_file("VAST_API_KEY")
    if key:
        return key
    sys.exit("VAST_API_KEY not found (env var or .env.vast)")


def _post_json(url: str, body: dict, headers: dict | None = None, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          # some proxies (RunPod/Cloudflare) 403 the default
                                          # "Python-urllib" identity — send a normal one
                                          "User-Agent": "si-gpu-masking-client/1.0",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _endpoint_key(account_key: str) -> str:
    """The /route/ service authenticates with the ENDPOINT's own key (rotating), not the
    account key — fetch it from the endpoint list."""
    req = urllib.request.Request("https://console.vast.ai/api/v0/endptjobs/",
                                 headers={"Authorization": f"Bearer {account_key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    for ep in data.get("results", []):
        if ep.get("endpoint_name") == ENDPOINT_NAME:
            return ep["api_key"]
    sys.exit(f"endpoint '{ENDPOINT_NAME}' not found in this account/team")


def call(route: str, payload: dict) -> dict:
    if DIRECT_URL:  # RunPod / plain host: straight POST, X-Api-Key auth
        t0 = time.perf_counter()
        headers = {"X-Api-Key": DIRECT_KEY} if DIRECT_KEY else {}
        out = _post_json(DIRECT_URL.rstrip("/") + route, payload, headers)
        print(f"direct {DIRECT_URL} answered in {round((time.perf_counter() - t0) * 1000)}ms "
              f"(server ms: {out.get('ms', '?')})")
        return out

    key = _endpoint_key(_api_key())
    t0 = time.perf_counter()
    auth = _post_json(ROUTE_URL, {"endpoint": ENDPOINT_NAME, "cost": 100},
                      {"Authorization": f"Bearer {key}"})
    if "url" not in auth:
        sys.exit(f"no worker available: {auth}")
    print(f"routed to {auth['url']} in {round((time.perf_counter() - t0) * 1000)}ms")
    t1 = time.perf_counter()
    body = {"auth_data": {k: auth[k] for k in ("signature", "cost", "endpoint", "reqnum", "url") if k in auth},
            "payload": payload}
    out = _post_json(auth["url"].rstrip("/") + route, body)
    print(f"worker answered in {round((time.perf_counter() - t1) * 1000)}ms "
          f"(server ms: {out.get('ms', '?')})")
    return out


def _b64_of(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode("ascii")


def _save_b64(b64: str, path: str):
    open(path, "wb").write(base64.b64decode(b64))
    print(f"wrote {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "health"
    if cmd == "health":
        # /health isn't a PyWorker route; run a remove on a 1x1 PNG as the health probe
        tiny_png_b64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
                        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
        print(json.dumps(call("/v1/remove", {"image_b64": tiny_png_b64}))[:200])
    elif cmd == "remove":
        out = call("/v1/remove", {"image_b64": _b64_of(sys.argv[2]), "feather": 1.0})
        _save_b64(out["image_b64"], "vast-cutout.png")
    elif cmd == "mask":
        out = call("/v1/mask", {"image_b64": _b64_of(sys.argv[2]), "variant": "soft"})
        _save_b64(out["mask_b64"], "vast-mask.png")
    elif cmd == "refine":
        x, y = int(sys.argv[3]), int(sys.argv[4])
        out = call("/v1/refine", {"image_b64": _b64_of(sys.argv[2]),
                                  "points": [{"x": x, "y": y, "label": 1}]})
        print(f"score: {out.get('score')}")
        _save_b64(out["mask_b64"], "vast-refine-mask.png")
    else:
        sys.exit(f"unknown command: {cmd}")
