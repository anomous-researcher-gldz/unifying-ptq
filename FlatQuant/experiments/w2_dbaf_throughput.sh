#!/bin/bash
# W2 Experiment: DBAF Deployment Feasibility
# Measures inference throughput (PPL eval time) with and without DBAF
# to show fold/unfold overhead is negligible.
#
# Approach: Both configs use the SAME trained matrices (with DBAF).
# At eval time we toggle DBAF on/off to isolate the runtime overhead
# of the fold/unfold ops — not the quality difference.
#
# Usage: bash experiments/w2_dbaf_throughput.sh

set -e
PYTHON=./venv_flatquant/bin/python
MODEL=./modelzoo/meta-llama/Meta-Llama-3-8B
MATRIX_PATH=./outputs/Meta-Llama-3-8B/w4a4/exp
OUTDIR=./outputs/experiments/w2_throughput
TASKS="arc_challenge arc_easy hellaswag lambada_openai piqa winogrande"

mkdir -p "$OUTDIR"

COMMON_ARGS="--model $MODEL \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --lwc --lac --add_diag \
    --reload_matrix --matrix_path $MATRIX_PATH \
    --lm_eval --lm_eval_batch_size 16 \
    --tasks $TASKS"

echo "============================================"
echo "W2 Experiment: DBAF Throughput Overhead"
echo "============================================"

# Run 1: WITH DBAF (default)
echo ""
echo ">>> Run 1/2: FlatQuant + DBAF + PCSA (full system)"
$PYTHON ./main.py $COMMON_ARGS \
    --output_dir "$OUTDIR/with_dbaf" \
    2>&1 | tee "$OUTDIR/with_dbaf.log"

# Run 2: WITHOUT DBAF
echo ""
echo ">>> Run 2/2: FlatQuant + PCSA only (DBAF disabled)"
$PYTHON ./main.py $COMMON_ARGS \
    --disable_dbaf \
    --output_dir "$OUTDIR/without_dbaf" \
    2>&1 | tee "$OUTDIR/without_dbaf.log"

echo ""
echo "============================================"
echo "Results saved to $OUTDIR"
echo "Compare wikitext2 PPL eval times and lm_eval accuracy between runs."
echo "============================================"
