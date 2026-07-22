#!/bin/bash
# Pod entrypoint: SSH support (RunPod injects the account's key as $PUBLIC_KEY) + the
# usual code-refresh + model server. Works on RunPod pods and any plain Docker host.

# --- sshd (only if a public key was provided and sshd exists) ---
if [ -n "$PUBLIC_KEY" ] && command -v sshd >/dev/null 2>&1; then
  mkdir -p /root/.ssh /run/sshd
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
  /usr/sbin/sshd -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no
  echo "sshd up (key auth)"
fi

# --- freshen worker code (code fixes ship with a pod restart, no rebuild) ---
cd /workspace/vast-pyworker && (git pull --ff-only || true)

# --- model server (API on 18000, guarded by API_KEY env) ---
exec python3 /workspace/vast-pyworker/server.py
