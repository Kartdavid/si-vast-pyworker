#!/bin/bash
# Baked bootstrap (image layer): freshen the repo, then hand over to the REPO's boot
# script — so all future boot-logic changes ship with a pod restart, never a rebuild.
cd /workspace/vast-pyworker && (git pull --ff-only || true)
if [ -f /workspace/vast-pyworker/start-pod.sh ]; then
  exec bash /workspace/vast-pyworker/start-pod.sh
fi
exec python3 /workspace/vast-pyworker/server.py
