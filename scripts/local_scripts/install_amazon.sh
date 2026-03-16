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
# Phase 2: Conda bootstrap
# -----------------------------------------------
echo "--- Phase 2: Conda bootstrap ---"

USE_CONDA=false
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    echo "Conda found: $(conda --version)"
    USE_CONDA=true
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    echo "Conda found at ~/miniconda3: $(conda --version)"
    USE_CONDA=true
else
    echo "Conda not found. Attempting to install Miniconda..."
    MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
    if wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$MINICONDA_INSTALLER" 2>/dev/null; then
        bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3" \
            || { echo "Miniconda installation failed" >&2; exit 1; }
        rm -f "$MINICONDA_INSTALLER"
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        echo "Miniconda installed: $(conda --version)"
        USE_CONDA=true
    else
        echo "Could not download Miniconda (no network?). Falling back to Python venv."
        rm -f "$MINICONDA_INSTALLER"
    fi
fi
echo ""

# -----------------------------------------------
# Phase 3: Create virtual environment in /tmp
# -----------------------------------------------
echo "--- Phase 3: Virtual environment ---"

mkdir -p /tmp/python-venv

if $USE_CONDA; then
    if [ -d "$ENV_DIR" ]; then
        echo "Conda env '$ENV_NAME' already exists at $ENV_DIR"
    else
        echo "Creating conda env '$ENV_NAME' at $ENV_DIR..."
        conda create --prefix "$ENV_DIR" python=3.11 -y \
            || { echo "conda create failed" >&2; exit 1; }
    fi
    conda activate "$ENV_DIR" || { echo "conda activate failed" >&2; exit 1; }
else
    if [ -d "$ENV_DIR" ]; then
        echo "Python venv '$ENV_NAME' already exists at $ENV_DIR"
    else
        echo "Creating Python venv '$ENV_NAME' at $ENV_DIR..."
        python3 -m venv "$ENV_DIR" \
            || { echo "python3 -m venv failed" >&2; exit 1; }
    fi
    source "$ENV_DIR/bin/activate" || { echo "venv activate failed" >&2; exit 1; }
fi

echo "Active Python: $(which python) ($(python --version))"
echo ""

# -----------------------------------------------
# Phase 4: Install from requirements.txt
# -----------------------------------------------
echo "--- Phase 4: Install Python dependencies ---"

pip install --upgrade pip || { echo "pip upgrade failed" >&2; exit 1; }

if [ -f "$REPO_ROOT/requirements.txt" ] && [ -s "$REPO_ROOT/requirements.txt" ]; then
    PYTHONNOUSERSITE=1 pip install -r "$REPO_ROOT/requirements.txt" \
        || { echo "pip install requirements failed" >&2; exit 1; }
else
    echo "WARNING: requirements.txt not found or empty at $REPO_ROOT" >&2
fi
echo ""

# -----------------------------------------------
# Phase 5: Verify GPU + PyTorch
# -----------------------------------------------
echo "--- Phase 5: GPU + PyTorch verification ---"

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}, VRAM: {props.total_mem / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available in PyTorch. GPU ops will fail until /dev/nvidia* is accessible.')
" || { echo "PyTorch verification failed" >&2; exit 1; }
echo ""

# -----------------------------------------------
# Phase 6: Download Gemma 2B model
# -----------------------------------------------
echo "--- Phase 6: Download model ---"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Model already exists at $MODEL_DIR, skipping download."
else
    echo "Downloading $MODEL_ID to $MODEL_DIR..."
    mkdir -p "$MODEL_DIR"
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
" || { echo "Model download failed. You may need to run: huggingface-cli login" >&2; exit 1; }
fi
echo ""

# -----------------------------------------------
# Phase 7: Register Jupyter kernel
# -----------------------------------------------
echo "--- Phase 7: Jupyter kernel ---"

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
if $USE_CONDA; then
    conda deactivate
else
    deactivate 2>/dev/null || true
fi

echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_DIR"
echo ""
echo "To run an experiment:"
echo "  python scripts/run_experiment.py --dataset agnews --epsilon 1.0"
echo ""
echo "To use in Jupyter, select kernel: Python (llm-scoring-env)"
