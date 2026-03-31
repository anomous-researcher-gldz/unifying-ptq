#!/bin/bash
# W5 Experiment: Calibration Dependency and Reproducibility
# Runs 2 configs to show robustness across seed + calibration dataset:
#   1. seed=0,  cali_dataset=c4       (different dataset, same seed as default)
#   2. seed=42, cali_dataset=wikitext2 (different seed, same dataset as default)
# Compare against existing default run: seed=0, cali_dataset=wikitext2
#
# Usage: bash experiments/w5_calibration_sensitivity.sh

set -e
PYTHON=./venv_flatquant/bin/python
MODEL=./modelzoo/meta-llama/Meta-Llama-3-8B
OUTDIR=./outputs/experiments/w5_sensitivity
TASKS="arc_challenge arc_easy hellaswag lambada_openai piqa winogrande"

mkdir -p "$OUTDIR"

COMMON_ARGS="--model $MODEL \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --save_matrix \
    --lm_eval --lm_eval_batch_size 16 \
    --tasks $TASKS"

echo "============================================"
echo "W5 Experiment: Calibration Sensitivity"
echo "Baseline (already run): seed=0, dataset=wikitext2"
echo "============================================"

# Run 1: seed=0, c4 (isolates dataset effect)
echo ""
echo ">>> Run 1/2: seed=0, dataset=c4"
$PYTHON ./main.py $COMMON_ARGS \
    --seed 0 \
    --cali_dataset c4 \
    --output_dir "$OUTDIR" \
    --exp_name "seed0_c4" \
    2>&1 | tee "$OUTDIR/seed0_c4.log"

# Run 2: seed=42, wikitext2 (isolates seed effect)
echo ""
echo ">>> Run 2/2: seed=42, dataset=wikitext2"
$PYTHON ./main.py $COMMON_ARGS \
    --seed 42 \
    --cali_dataset wikitext2 \
    --output_dir "$OUTDIR" \
    --exp_name "seed42_wikitext2" \
    2>&1 | tee "$OUTDIR/seed42_wikitext2.log"

echo ""
echo "============================================"
echo "Done. Compare against baseline (seed=0, wikitext2, acc_avg=70.38):"
echo "  grep -h 'acc_avg' $OUTDIR/*.log"
echo "============================================"
