#!/bin/bash
# Pod boot (REPO-owned — editable via git push + pod restart, no image rebuild).
# 1) SSH: RunPod injects the account key as $PUBLIC_KEY. sshd needs HOST keys too —
#    `ssh-keygen -A` generates any that are missing (their absence = silent sshd death
#    = "connection refused"; learned 22 Jul).
if [ -n "$PUBLIC_KEY" ] && command -v sshd >/dev/null 2>&1; then
  mkdir -p /root/.ssh /run/sshd
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
  ssh-keygen -A
  /usr/sbin/sshd -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no \
    && echo "sshd up (key auth)" || echo "sshd FAILED to start"
fi

# 2) Model server (API on 18000, guarded by API_KEY env)
exec python3 /workspace/vast-pyworker/server.py
