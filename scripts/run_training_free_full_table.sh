#!/usr/bin/env bash
# Training-free full table sweep: RTN/GPTQ/AWQ ± {DBAF, PCSA-tf, both} × all models.
#
# Env-var overrides:
#   TARGETS  — space- or comma-separated list (default: all 8 models)
#   METHODS  — space- or comma-separated list (default: rtn gptq awq)
#   AUGMENTS — space- or comma-separated list (default: alone dbaf pcsa_tf dbaf+pcsa_tf)
#   OUT_BASE — output root dir              (default: /data/outputs/G8-training-free-full)
#   CONDA_ENV_NAME — conda env to activate  (default: unifyptq)
#
# Usage examples:
#   bash scripts/run_training_free_full_table.sh
#   TARGETS=llama3-8b,qwen25-7b METHODS=rtn AUGMENTS=alone,dbaf bash scripts/run_training_free_full_table.sh
#   TARGETS=swinir-x2 METHODS=rtn,gptq AUGMENTS=alone,dbaf,pcsa_tf,dbaf+pcsa_tf bash scripts/run_training_free_full_table.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DRIVER="$SCRIPT_DIR/run_training_free_full_table.py"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-unifyptq}"
OUT_BASE="${OUT_BASE:-/data/outputs/G8-training-free-full}"

# Default lists (comma-separated env vars, or space-separated here)
_DEFAULT_TARGETS="llama3-8b qwen25-7b sam-b sam-l sam-h swinir-x2 swinir-x3 swinir-x4"
_DEFAULT_METHODS="rtn gptq awq"
_DEFAULT_AUGMENTS="alone dbaf pcsa_tf dbaf+pcsa_tf"

# Convert env-var overrides (comma → space) or use defaults
if [[ -n "${TARGETS:-}" ]]; then
    TARGETS_LIST="${TARGETS//,/ }"
else
    TARGETS_LIST="$_DEFAULT_TARGETS"
fi

if [[ -n "${METHODS:-}" ]]; then
    METHODS_LIST="${METHODS//,/ }"
else
    METHODS_LIST="$_DEFAULT_METHODS"
fi

if [[ -n "${AUGMENTS:-}" ]]; then
    AUGMENTS_LIST="${AUGMENTS//,/ }"
else
    AUGMENTS_LIST="$_DEFAULT_AUGMENTS"
fi

# ---------------------------------------------------------------------------
# Conda activation helper (works both inside and outside conda init)
# ---------------------------------------------------------------------------
_activate_conda() {
    local env_name="$1"
    # Try standard conda init path; fall back to direct activate
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "$env_name" 2>/dev/null || true
    fi
    # If CONDA_PREFIX already set to the right env, nothing to do
    if [[ "${CONDA_PREFIX:-}" == *"$env_name"* ]]; then
        return 0
    fi
    # Try direct path activation
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
    if [[ -f "$conda_base/envs/$env_name/bin/activate" ]]; then
        source "$conda_base/envs/$env_name/bin/activate"
    fi
}

_activate_conda "$CONDA_ENV_NAME"
PYTHON="${CONDA_PREFIX:-}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(which python3 || which python)"
fi
echo "[sweep] Using python: $PYTHON"
echo "[sweep] targets : $TARGETS_LIST"
echo "[sweep] methods : $METHODS_LIST"
echo "[sweep] augments: $AUGMENTS_LIST"
echo "[sweep] out_base: $OUT_BASE"
echo ""

# ---------------------------------------------------------------------------
# Summary accounting
# ---------------------------------------------------------------------------
N_TOTAL=0
N_SKIP=0
N_OK=0
N_FAIL=0
declare -a FAILED_CELLS=()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for TARGET in $TARGETS_LIST; do
    for METHOD in $METHODS_LIST; do
        for AUGMENT in $AUGMENTS_LIST; do

            N_TOTAL=$((N_TOTAL + 1))
            CELL_NAME="${METHOD}_${AUGMENT}"
            OUT_DIR="$OUT_BASE/$TARGET/$CELL_NAME"
            OUT_JSON="$OUT_DIR/eval.json"
            LOG_FILE="$OUT_DIR/eval.log"

            # ------------------------------------------------------------------
            # Skip check
            # ------------------------------------------------------------------
            if [[ -f "$OUT_JSON" ]]; then
                echo "[sweep] SKIP  target=$TARGET  method=$METHOD  augments=$AUGMENT  (exists: $OUT_JSON)"
                N_SKIP=$((N_SKIP + 1))
                continue
            fi

            # ------------------------------------------------------------------
            # Create output directory and log
            # ------------------------------------------------------------------
            mkdir -p "$OUT_DIR"

            echo "[sweep] RUN   target=$TARGET  method=$METHOD  augments=$AUGMENT"
            echo "[sweep]       -> $OUT_JSON"
            echo "[sweep]       -> log: $LOG_FILE"

            # ------------------------------------------------------------------
            # Run the Python driver, tee to log
            # ------------------------------------------------------------------
            set +e
            "$PYTHON" "$DRIVER" \
                --target "$TARGET" \
                --method "$METHOD" \
                --augments "$AUGMENT" \
                --out "$OUT_JSON" \
                2>&1 | tee "$LOG_FILE"
            EXIT_CODE="${PIPESTATUS[0]}"
            set -e

            if [[ $EXIT_CODE -eq 0 && -f "$OUT_JSON" ]]; then
                N_OK=$((N_OK + 1))
                # Print one-line summary from JSON
                PPL_WT2=$(python3 -c "import json,sys; d=json.load(open('$OUT_JSON')); m=d.get('metrics',{}); print(m.get('wikitext2_ppl','n/a'))" 2>/dev/null || echo "n/a")
                PPL_C4=$(python3  -c "import json,sys; d=json.load(open('$OUT_JSON')); m=d.get('metrics',{}); print(m.get('c4_ppl','n/a'))" 2>/dev/null || echo "n/a")
                CMAP=$(python3    -c "import json,sys; d=json.load(open('$OUT_JSON')); m=d.get('metrics',{}); print(m.get('coco_map','n/a'))" 2>/dev/null || echo "n/a")
                S5=$(python3      -c "import json,sys; d=json.load(open('$OUT_JSON')); m=d.get('metrics',{}); print(m.get('set5_psnr_db','n/a'))" 2>/dev/null || echo "n/a")
                U100=$(python3    -c "import json,sys; d=json.load(open('$OUT_JSON')); m=d.get('metrics',{}); print(m.get('urban100_psnr_db','n/a'))" 2>/dev/null || echo "n/a")
                echo "[sweep] OK    target=$TARGET method=$METHOD augments=$AUGMENT | wt2_ppl=$PPL_WT2 c4_ppl=$PPL_C4 coco_map=$CMAP set5=$S5 urban100=$U100"
            else
                N_FAIL=$((N_FAIL + 1))
                FAILED_CELLS+=("$TARGET/$METHOD/$AUGMENT")
                echo "[sweep] FAIL  target=$TARGET  method=$METHOD  augments=$AUGMENT  (exit=$EXIT_CODE)"
            fi
            echo ""

        done
    done
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "[sweep] DONE  total=$N_TOTAL  skip=$N_SKIP  ok=$N_OK  fail=$N_FAIL"
if [[ ${#FAILED_CELLS[@]} -gt 0 ]]; then
    echo "[sweep] FAILED cells:"
    for cell in "${FAILED_CELLS[@]}"; do
        echo "  - $cell"
    done
fi
echo "======================================================================"
