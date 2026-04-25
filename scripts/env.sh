# Per-shell activation for the AURA-1 project on RunPod.
#
# Sourced from ~/.bashrc by scripts/setup-pod.sh, so every new interactive
# shell (including new tmux panes) gets the venv and HF cache wired up
# without the user typing anything.
#
# Both targets live on /workspace (the persistent network volume), so the
# venv and downloaded model weights survive pod migrations. ~/.bashrc itself
# is on the container's ephemeral fs and is wiped on migration — that's what
# setup-pod.sh re-establishes.

# Activate the project venv. Always re-source rather than guarding on
# VIRTUAL_ENV — RunPod's /etc/rp_environment (sourced from ~/.bashrc just
# before this file) resets PATH to the system default, which strips the
# venv bin dir even though the inherited VIRTUAL_ENV var still claims the
# venv is active. Re-sourcing activate puts /workspace/venv/bin back at
# the front of PATH; it's idempotent so running it twice is harmless.
if [ -f /workspace/venv/bin/activate ]; then
    source /workspace/venv/bin/activate
fi

# Point the Hugging Face cache at /workspace so the ~16GB Qwen2.5-VL weights
# survive pod migrations.
export HF_HOME=/workspace/.cache/huggingface
