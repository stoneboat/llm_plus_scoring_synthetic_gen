#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="llm_scoring_venv"
ENV_DIR="/tmp/python-venv/$ENV_NAME"
MODEL_ID="google/gemma-2-2b-it"
MODEL_DIR="$REPO_ROOT/data/models/gemma-2-2b-it"

echo "============================================"
echo "  LLM + Scoring Synthetic Gen — Amazon Setup"
echo "============================================"
echo "Repo root: $REPO_ROOT"
echo ""

# -----------------------------------------------
# Phase 0: Disable IPv6 (prevents HuggingFace hangs on EC2)
# -----------------------------------------------
echo "--- Phase 0: Network fix (disable IPv6) ---"
if sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1 &>/dev/null \
   && sudo sysctl -w net.ipv6.conf.default.disable_ipv6=1 &>/dev/null; then
    echo "IPv6 disabled — HuggingFace requests will use IPv4."
else
    echo "WARNING: Could not disable IPv6 (no sudo?). HuggingFace downloads may hang."
fi
echo ""

# -----------------------------------------------
# Phase 1: GPU setup (Amazon-specific)
# -----------------------------------------------
echo "--- Phase 1: GPU setup ---"

GPU_OK=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_OK=true
else
    echo "nvidia-smi not working, attempting to load NVIDIA kernel module..."
    sudo modprobe nvidia 2>/dev/null || modprobe nvidia 2>/dev/null || true
    sudo modprobe nvidia_uvm 2>/dev/null || modprobe nvidia_uvm 2>/dev/null || true
    # Try nvidia-modprobe to create /dev/nvidia* device files
    command -v nvidia-modprobe &>/dev/null && nvidia-modprobe 2>/dev/null || true
    if nvidia-smi &>/dev/null; then
        GPU_OK=true
    fi
fi

if $GPU_OK; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not available. GPU may not be accessible from this shell."
    echo "  The NVIDIA kernel modules may be loaded (check: lsmod | grep nvidia)"
    echo "  but /dev/nvidia* device files may be missing (e.g. sandbox restriction)."
    echo "  Continuing with CPU-only setup. Re-run from an unsandboxed shell for GPU."
    echo ""
fi
echo ""

# -----------------------------------------------
# Phase 2: Create virtual environment in /tmp
# -----------------------------------------------
echo "--- Phase 2: Virtual environment ---"

mkdir -p /tmp/python-venv

# If the directory exists but activate is missing, it's a broken venv — remove and recreate
if [ -d "$ENV_DIR" ] && [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "Detected broken venv at $ENV_DIR (missing activate). Removing and recreating..."
    rm -rf "$ENV_DIR"
fi

if [ -d "$ENV_DIR" ]; then
    echo "Python venv '$ENV_NAME' already exists at $ENV_DIR"
else
    echo "Creating Python venv '$ENV_NAME' at $ENV_DIR..."
    # --without-pip avoids the ensurepip dependency (broken on held python3 packages)
    python3 -m venv --without-pip "$ENV_DIR" \
        || { echo "python3 -m venv failed" >&2; exit 1; }

    # Bootstrap pip manually via get-pip.py
    echo "Bootstrapping pip via get-pip.py..."
    GET_PIP="/tmp/get-pip.py"
    wget -q https://bootstrap.pypa.io/get-pip.py -O "$GET_PIP" \
        || { echo "Failed to download get-pip.py" >&2; exit 1; }
    "$ENV_DIR/bin/python3" "$GET_PIP" \
        || { echo "get-pip.py failed" >&2; exit 1; }
    rm -f "$GET_PIP"
fi
source "$ENV_DIR/bin/activate" || { echo "venv activate failed" >&2; exit 1; }

echo "Active Python: $(which python) ($(python --version))"
echo ""

# -----------------------------------------------
# Phase 3: Install from requirements.txt
# -----------------------------------------------
echo "--- Phase 3: Install Python dependencies ---"

pip install --upgrade pip || { echo "pip upgrade failed" >&2; exit 1; }

if [ -f "$REPO_ROOT/requirements.txt" ] && [ -s "$REPO_ROOT/requirements.txt" ]; then
    PYTHONNOUSERSITE=1 pip install -r "$REPO_ROOT/requirements.txt" \
        || { echo "pip install requirements failed" >&2; exit 1; }
else
    echo "WARNING: requirements.txt not found or empty at $REPO_ROOT" >&2
fi

# Ensure a recent huggingface_hub (older versions lack the CLI and login API)
pip install -U "huggingface_hub>=0.23.0" \
    || { echo "huggingface_hub upgrade failed" >&2; exit 1; }
echo ""

# -----------------------------------------------
# Phase 4: Verify GPU + PyTorch
# -----------------------------------------------
echo "--- Phase 4: GPU + PyTorch verification ---"

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available in PyTorch. GPU ops will fail until /dev/nvidia* is accessible.')
" || { echo "PyTorch verification failed" >&2; exit 1; }
echo ""

# -----------------------------------------------
# Phase 5: Download Gemma 2B model
# -----------------------------------------------
echo "--- Phase 5: Download model ---"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Model already exists at $MODEL_DIR, skipping download."
else
    echo "Downloading $MODEL_ID to $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"

    # Authenticate if HF_TOKEN is set; otherwise rely on cached credentials
    if [ -n "$HF_TOKEN" ]; then
        echo "HF_TOKEN found — logging in to HuggingFace..."
        python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" \
            || { echo "HuggingFace login failed" >&2; exit 1; }
    elif [ ! -f "$HOME/.cache/huggingface/token" ] && [ ! -f "$HOME/.huggingface/token" ]; then
        echo "" >&2
        echo "ERROR: Gemma is a gated model and no credentials were found." >&2
        echo "Please authenticate first, then re-run this script:" >&2
        echo "" >&2
        echo "  Option A — environment variable (recommended for automation):" >&2
        echo "    export HF_TOKEN=hf_YOUR_TOKEN" >&2
        echo "    bash $0" >&2
        echo "" >&2
        echo "  Option B — interactive login:" >&2
        echo "    source $ENV_DIR/bin/activate" >&2
        echo "    python -c \"from huggingface_hub import login; login(token='hf_YOUR_TOKEN')\"" >&2
        echo "" >&2
        echo "Get a token at: https://huggingface.co/settings/tokens" >&2
        echo "Accept the license at: https://huggingface.co/google/gemma-2-2b-it" >&2
        exit 1
    fi

    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
model_id = '$MODEL_ID'
save_dir = '$MODEL_DIR'
print(f'Downloading tokenizer from {model_id}...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir)
print(f'Downloading model from {model_id}...')
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto')
model.save_pretrained(save_dir)
print(f'Model saved to {save_dir}')
" || { echo "Model download failed." >&2; exit 1; }
fi
echo ""

# -----------------------------------------------
# Phase 6: Register Jupyter kernel
# -----------------------------------------------
echo "--- Phase 6: Jupyter kernel ---"

python -m ipykernel uninstall --user --name=llm-scoring-env -y 2>/dev/null || true

python -m ipykernel install --user \
    --name=llm-scoring-env \
    --display-name "Python (llm-scoring-env)" \
    || { echo "ipykernel install failed" >&2; exit 1; }

KERNEL_DIR="$HOME/.local/share/jupyter/kernels/llm-scoring-env"
if [ -d "$KERNEL_DIR" ]; then
    cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$ENV_DIR/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (llm-scoring-env)",
  "language": "python"
}
EOL
    echo "Jupyter kernel registered at $KERNEL_DIR"
fi
echo ""

# -----------------------------------------------
# Done
# -----------------------------------------------
deactivate 2>/dev/null || true

echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "To run an experiment:"
echo "  python scripts/run_experiment.py --dataset agnews --epsilon 1.0"
echo ""
echo "To use in Jupyter, select kernel: Python (llm-scoring-env)"
