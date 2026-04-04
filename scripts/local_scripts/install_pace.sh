#!/bin/bash
module purge
module load pytorch/25
module load anaconda3

# Initialize conda in this non-interactive shell
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found after loading anaconda3 module" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="llm_scoring_venv"
ENV_DIR="/tmp/python-venv/$ENV_NAME"
MODEL_ID="google/gemma-2-2b-it"
MODEL_DIR="$REPO_ROOT/data/models/gemma-2-2b-it"

# Hugging Face Hub downloads default to ~/.cache/huggingface; PACE home quotas are often too small for ~5GB+ models.
# If you prefer another disk (e.g. export HF_HOME="$SCRATCH/.hf" before running), set HF_HOME / HF_HUB_CACHE yourself.
if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="$REPO_ROOT/data/.hf_home"
    mkdir -p "$HF_HOME"
    echo "Using HF_HOME=$HF_HOME for Hub cache (set HF_HOME to override)."
fi

mkdir -p /tmp/python-venv

if [ -d "$ENV_DIR" ]; then
    echo "Conda env '$ENV_NAME' already exists in $ENV_DIR."
else
    echo "Creating conda env '$ENV_NAME' in $ENV_DIR..."
    conda create --prefix "$ENV_DIR" python=3.11 -y || { echo "conda create failed" >&2; exit 1; }
fi

# Activate the conda environment
conda activate "$ENV_DIR" || { echo "conda activate failed" >&2; exit 1; }

# Use env binaries explicitly — module-loaded python on PATH can differ from where pip installs.
PY="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"
hash -r

# Upgrade pip
"$PIP" install --upgrade pip || { echo "pip upgrade failed" >&2; exit 1; }

# Install requirements.txt
if [ -f "$REPO_ROOT/requirements.txt" ] && [ -s "$REPO_ROOT/requirements.txt" ]; then
    PYTHONNOUSERSITE=1 "$PIP" install -r "$REPO_ROOT/requirements.txt" || { echo "pip install requirements failed" >&2; exit 1; }
fi

# Ensure a recent huggingface_hub
"$PIP" install -U "huggingface_hub>=0.23.0" \
    || { echo "huggingface_hub upgrade failed" >&2; exit 1; }
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
        "$PY" -c "from huggingface_hub import login; login(token='$HF_TOKEN')" \
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
        echo "    conda activate $ENV_DIR" >&2
        echo "    $PY -c \"from huggingface_hub import login; login(token='hf_YOUR_TOKEN')\"" >&2
        echo "" >&2
        echo "Get a token at: https://huggingface.co/settings/tokens" >&2
        echo "Accept the license at: https://huggingface.co/google/gemma-2-2b-it" >&2
        exit 1
    fi

    "$PY" -c "
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
# Phase 6: Jupyter kernel
# -----------------------------------------------
echo "--- Phase 6: Jupyter kernel ---"

# Uninstall existing kernel if it exists (to avoid conflicts)
"$PY" -m ipykernel uninstall --user --name=llm-scoring-env -y 2>/dev/null || true

# Register kernel first (this creates the directory and basic kernel.json)
"$PY" -m ipykernel install --user --name=llm-scoring-env --display-name "Python (llm-scoring-env)" || { echo "ipykernel install failed" >&2; exit 1; }

# Create the custom kernel spec directory (in case ipykernel didn't create it)
KERNEL_DIR=~/.local/share/jupyter/kernels/llm-scoring-env
mkdir -p "$KERNEL_DIR"

# Create a simple wrapper that loads modules before starting kernel
KERNEL_WRAPPER="$KERNEL_DIR/kernel_wrapper.sh"
LOG_FILE="$KERNEL_DIR/kernel_wrapper.log"
cat > "$KERNEL_WRAPPER" <<'WRAPPER_EOF'
#!/bin/bash
# Simple wrapper to load modules before starting kernel
# The Python executable is already specified in kernel.json, so we just need to:
# 1. Load pytorch/25 module (provides torch)
# 2. Set up PYTHONPATH so Python can find torch

# Log wrapper invocation
LOG_FILE="$HOME/.local/share/jupyter/kernels/llm-scoring-env/kernel_wrapper.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') kernel_wrapper invoked with args: $*" >> "$LOG_FILE"

# Initialize module system
if [ -f /usr/local/pace-apps/lmod/lmod/init/bash ]; then
    source /usr/local/pace-apps/lmod/lmod/init/bash
fi

# Load pytorch/25 module
module load pytorch/25 2>/dev/null

# Reduce CUDA allocator fragmentation for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get Python version from the Python executable that will be used (from kernel.json)
# The first argument after the wrapper should be the Python executable path
PYTHON_EXE="$1"
if [ -n "$PYTHON_EXE" ] && [ -f "$PYTHON_EXE" ]; then
    PYTHON_VERSION=$("$PYTHON_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    
    # If torch is not accessible, set up PYTHONPATH
    if ! "$PYTHON_EXE" -c "import torch" 2>/dev/null; then
        # Add Python dist-packages to PYTHONPATH (where pytorch module installs torch)
        PYTHON_DIST_PACKAGES="/usr/local/lib/python${PYTHON_VERSION}/dist-packages"
        PYTHON_DIST_PACKAGES64="/usr/local/lib64/python${PYTHON_VERSION}/dist-packages"
        
        # Build PYTHONPATH with both paths
        NEW_PYTHONPATH=""
        [ -d "$PYTHON_DIST_PACKAGES" ] && NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES"
        [ -d "$PYTHON_DIST_PACKAGES64" ] && {
            [ -n "$NEW_PYTHONPATH" ] && NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES64:$NEW_PYTHONPATH" || NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES64"
        }
        
        # If directories don't exist, add them anyway (module might set them up dynamically)
        [ -z "$NEW_PYTHONPATH" ] && NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES64:$PYTHON_DIST_PACKAGES"
        
        # Add to existing PYTHONPATH or set new one
        [ -z "$PYTHONPATH" ] && export PYTHONPATH="$NEW_PYTHONPATH" || export PYTHONPATH="$NEW_PYTHONPATH:$PYTHONPATH"
    fi
fi

# Execute the python command with all arguments
[ -n "$PYTHONPATH" ] && exec env PYTHONPATH="$PYTHONPATH" "$@" || exec "$@"
WRAPPER_EOF
chmod +x "$KERNEL_WRAPPER"

# Write the kernel.json with wrapper (use absolute path for wrapper)
KERNEL_WRAPPER_ABS=$(readlink -f "$KERNEL_WRAPPER" 2>/dev/null || echo "$KERNEL_WRAPPER")
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$KERNEL_WRAPPER_ABS",
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

# Verify kernel registration
if [ ! -f "$KERNEL_DIR/kernel.json" ]; then
    echo "Error: kernel.json was not created properly" >&2
    exit 1
fi
if [ ! -x "$KERNEL_WRAPPER" ]; then
    echo "Error: kernel_wrapper.sh is not executable" >&2
    exit 1
fi

conda deactivate

echo ""
echo "✅ Installation complete!"
echo "⚠️  Restart your Jupyter kernel to use the new environment in notebooks."
echo "To activate the environment: conda activate $ENV_DIR"
echo "To run an experiment: python scripts/run_experiment.py --dataset agnews --epsilon 1.0"
echo "To use in Jupyter, select kernel: Python (llm-scoring-env)"
