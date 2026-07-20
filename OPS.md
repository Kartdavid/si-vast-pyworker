# Ops — how this deploys

## Architecture (since 19 Jul 2026)

Everything runs from a **baked Docker image**: `ghcr.io/kartdavid/si-gpu-masking:latest`
(Python deps pinned, BiRefNet + SAM 2.1 weights included, worker code cloned in).
A fresh Vast machine only pulls the image → boot-to-Ready ≈ 3 minutes, no downloads,
no version drift. The build fails in CI if model loading ever breaks (sanity gate).

- **Code change** (server.py / worker.py): push to main → on-start `git pull` picks it up
  on the next worker start. No rebuild needed.
- **Dependency or model change** (requirements.txt / Dockerfile): push to main → the
  GitHub Action rebuilds and pushes the image (~20 min) → recycle workers.

## Vast template settings (template: "Si GPU masking (Serverless)")

- Image Path:Tag: `ghcr.io/kartdavid/si-gpu-masking:latest`
- Docker Options: `-p 3000:3000`
- Environment variables:
  - `SERVERLESS` = `true`
  - `USE_SYSTEM_PYTHON` = `true`   ← REQUIRED: use the image's Python (no venv/pip at boot)
  - `HF_HOME` = `/models`          ← models live here in the image
  - `MODEL_LOG` = `/var/log/model/server.log`
  - `MODEL_LOG_FILE` = `/var/log/model/server.log`
  - `MODEL_SERVER_URL` = `http://127.0.0.1:18000`
  - `MODEL_HEALTH_ENDPOINT` = `http://127.0.0.1:18000/health`
  - (PYWORKER_REPO/PYWORKER_REF no longer used — code is baked + git-pulled)
- On-start Script:
  ```
  (cd /workspace/vast-pyworker && git pull --ff-only) || true
  bootstrap_script=https://raw.githubusercontent.com/vast-ai/pyworker/refs/heads/main/start_server.sh
  curl -L "$bootstrap_script" | bash
  ```
- Extra Filters: `verified=true direct_port_count>=2 compute_cap>=800 gpu_ram>=16000 inet_down>=500 num_gpus=1 dph_total<=0.18`
- Disk: 40 GB

## One-time after the first image build

The GHCR package starts private. Make it public so Vast hosts can pull it:
github.com → Kartdavid org → Packages → `si-gpu-masking` → Package settings →
Danger Zone → Change visibility → Public.

## Endpoint / workergroup

Endpoint `si-gpu-masking-staging` (ID 31000): min workers 1, max 3, min load 1
(always-warm), no scale-to-zero. After template changes, delete + recreate the
workergroup so it picks up the new template.

## Smoke test (from the Mac)

```sh
cd ~/Developer/si-vast-pyworker
python3 test_client.py health
python3 test_client.py remove ~/Developer/stickerit-platform/apps/bg-remove/test-cutout.png
```
