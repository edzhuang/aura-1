#!/bin/bash
# One-shot recovery script for after a RunPod pod migration.
#
# RunPod's persistent volume is mounted at /workspace and survives migrations.
# Everything else (apt-installed tools, ~/.bashrc, tmux sessions, the HF token
# in ~/.cache/huggingface unless we redirect it) lives on the container's
# overlay fs and gets wiped when the pod moves to a new host.
#
# This script re-establishes the ephemeral state so you don't have to remember
# the steps each time. It's idempotent — safe to re-run.
#
# Usage (from anywhere on the pod):
#     bash /workspace/aura-1/scripts/setup-pod.sh
# Then open a new shell (or `source ~/.bashrc`) for the env to activate.

set -euo pipefail

REPO_ENV_SH="/workspace/aura-1/scripts/env.sh"
BASHRC_LINE="source ${REPO_ENV_SH}"

echo "[1/3] Installing tmux + htop (lost with container fs)..."
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux htop

echo "[2/3] Wiring ~/.bashrc to auto-source ${REPO_ENV_SH}..."
touch ~/.bashrc
if ! grep -qxF "${BASHRC_LINE}" ~/.bashrc; then
    {
        echo ""
        echo "# AURA-1: activate venv + HF cache on every new interactive shell."
        echo "${BASHRC_LINE}"
    } >> ~/.bashrc
    echo "  added to ~/.bashrc"
else
    echo "  already present in ~/.bashrc"
fi

echo "[3/3] Verifying the env activates cleanly..."
# Use `bash -i -c` so .bashrc is sourced exactly the way an interactive shell
# (or new tmux pane) would source it. If this fails, the next tmux pane fails.
bash -i -c '
    set -e
    [ "$(which python)" = "/workspace/venv/bin/python" ] || { echo "VENV NOT ACTIVE: which python = $(which python)"; exit 1; }
    [ "$HF_HOME" = "/workspace/.cache/huggingface" ] || { echo "HF_HOME WRONG: $HF_HOME"; exit 1; }
    echo "  python: $(which python)"
    echo "  HF_HOME: $HF_HOME"
'

echo
echo "Pod setup complete. Open a new shell or run \`source ~/.bashrc\` to activate."
