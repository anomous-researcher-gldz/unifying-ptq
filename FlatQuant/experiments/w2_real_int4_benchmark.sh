#!/bin/bash
# W2 Experiment: Real INT4 DBAF Throughput Overhead
# Uses the deploy/ CUDA kernels (CUTLASS INT4 GEMM) to measure
# actual prefill/decode latency with and without DBAF fold/unfold.
#
# This directly addresses the reviewer's concern about DBAF disrupting
# INT4 GEMM kernel fusion. The GEMM kernel is unchanged — DBAF ops
# happen on the activation side before/after the kernel.
#
# Requires: quantized model weights at ./outputs/Meta-Llama-3-8B/w4a4/exp/
#
# Usage: bash experiments/w2_real_int4_benchmark.sh

set -e
PYTHON=./venv_flatquant/bin/python
OUTDIR=./outputs/experiments/w2_real_int4

mkdir -p "$OUTDIR"

echo "============================================"
echo "W2: Real INT4 Benchmark — DBAF Overhead"
echo "============================================"

# Run 1: WITHOUT DBAF (baseline, dbaf_alpha not set → None)
echo ""
echo ">>> Run 1/2: Real INT4 FlatQuant (no DBAF)"
$PYTHON benchmarks/benchmark_model.py \
    --batch_size 1 --random_mode \
    --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
    2>&1 | tee "$OUTDIR/no_dbaf.log"

# Run 2: WITH DBAF (dbaf_alpha=0.99)
echo ""
echo ">>> Run 2/2: Real INT4 FlatQuant + DBAF (alpha=0.99)"
$PYTHON benchmarks/benchmark_model.py \
    --batch_size 1 --random_mode \
    --model ./modelzoo/meta-llama/Meta-Llama-3-8B \
    --dbaf_alpha 0.99 \
    2>&1 | tee "$OUTDIR/with_dbaf.log"

echo ""
echo "============================================"
echo "Results saved to $OUTDIR"
echo "Compare prefill/decode/e2e latency between runs."
echo "============================================"
