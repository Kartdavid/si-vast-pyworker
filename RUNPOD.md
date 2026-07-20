# RunPod deploy — beginner runbook (production GPU masking)

Goal: our baked image running on a **dedicated datacenter GPU** at RunPod, always on,
under $200/mo. No serverless machinery — the platform calls the pod's HTTPS URL directly,
exactly like it calls Modal. Modal stays wired as automatic fallback.

Total hands-on time: ~30 minutes (plus one ~20-min image rebuild wait).

---

## Step 0 — Push the updated code (GitHub Desktop, `si-vast-pyworker` repo ONLY)

Changed files: `server.py` (API-key auth + listen address), `Dockerfile` (default start
command), `test_client.py` (direct mode), `RUNPOD.md` (this file).

Commit summary: `Direct-serve mode for RunPod (auth, CMD, test client)` → **Push origin**.

Then: github.com/Kartdavid/si-vast-pyworker → **Actions** → wait for the green tick
(~20 min). That publishes the updated image to ghcr.

## Step 1 — RunPod account

1. Go to **runpod.io** → Sign Up (use david@stickerit.co).
2. Console → **Billing** → add a card and load **$25** one-time credit (enough for a
   week of testing; set up auto-pay later only once we're happy).

## Step 2 — Make an API key for our service

This is OUR key (the pod's front door), not a RunPod key. In Terminal:

```sh
openssl rand -hex 24
```

Copy the output into a note — you'll paste it twice below.

## Step 3 — Deploy the pod

1. RunPod Console → **Pods** → **Deploy**.
2. **GPU:** pick **RTX A5000** (24 GB) — Secure Cloud, ~$0.27/hr ≈ **$197/mo**.
   (If offered cheaper Community A5000 and you want to save ~$70/mo, that's fine too —
   still far more reliable than Vast.)
3. **Instance type:** On-Demand (NOT Spot/Interruptible — spot machines get taken away).
4. Click **Edit Template** (or "Customize deployment"):
   - **Container Image:** `ghcr.io/kartdavid/si-gpu-masking:latest`
   - **Container Disk:** 30 GB
   - **Expose HTTP Ports:** `18000`
   - **Environment Variables:** add one:
     - `API_KEY` = (paste the key from Step 2)
   - Leave Docker command empty (the image's default starts the server).
5. **Deploy On-Demand**.

The pod pulls the image (~13 GB, a few minutes in their datacenter) then shows **Running**.

## Step 4 — Get the URL

On the pod card → **Connect** → under HTTP services you'll see port **18000** with a link
like:

```
https://<pod-id>-18000.proxy.runpod.net
```

Copy it. Quick browser check: open `<that URL>/health` — you should see JSON with
`"ok": true` and, once the models finish warming (~1 min after boot), `"loaded": true`.

## Step 5 — Point the test client at it

Open the key file 📋:
```sh
open -e ~/Developer/si-vast-pyworker/.env.vast
```
Add two lines (paste your real values), then save:
```
GPU_MASK_URL=https://<pod-id>-18000.proxy.runpod.net
GPU_MASK_KEY=<the key from Step 2>
```

Smoke test 📋:
```sh
cd ~/Developer/si-vast-pyworker
python3 test_client.py health
python3 test_client.py remove ~/Developer/stickerit-platform/apps/bg-remove/test-cutout.png && open vast-cutout.png
python3 test_client.py refine ~/Developer/stickerit-platform/apps/bg-remove/test-cutout.png 350 350 && open vast-refine-mask.png
```

Success: `"ok": true`, a clean transparent cutout, and a SAM 2 mask — with ~1s server times.

## Step 6 — Cost guardrails

- **Billing page:** your credit balance IS the hard stop (no auto-pay = can't overspend).
- The pod bills per second while running: A5000 Secure ≈ $6.50/day.
- Stopping the pod (■ on the pod card) stops GPU billing (small disk fee remains).

## What happens next (Claude's side, no action needed yet)

- Wire the platform worker: `BGREMOVE_URL`/`BGREMOVE_KEY` pointing at the pod, Modal as
  automatic fallback → staging test via `?mode=background-remove` → benchmark vs Modal →
  production cutover.

## Ops notes

- **Code update:** push to si-vast-pyworker main → image rebuilds (~20 min) → on the pod
  card press ⟳ (restart with newest image). Zero-drama redeploys.
- **Region:** pick an EU datacenter if offered during deploy (nice-to-have, not critical).
- **If the pod ever misbehaves:** its logs are on the pod card → Logs — our server prints
  `MODELS_LOADED` / `MODEL_SERVER_READY` and per-request errors there.
