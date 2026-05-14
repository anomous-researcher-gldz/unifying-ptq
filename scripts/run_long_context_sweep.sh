#!/usr/bin/env bash
# Item 4 — Long-context regime micro-benchmark.
#
# Reuses scripts/micro_benchmark_primitives.py at a range of seq_lens to show
# how rotation cost compounds linearly with sequence length while DBAF/PCSA-tf
# scale the same way but with a smaller constant.
#
# Output:
#   scripts/_out/long_context/seq_{4K, 16K, 32K, 65K}_d{4096, 8192}.json
#
# Uses d=4096 (Llama-3-8B) and d=8192 (Llama-3-70B-class), keeping n_blocks=8
# so the GPU memory budget fits at seq=65K.
set -uo pipefail
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate unifyptq
mkdir -p scripts/_out/long_context

for D in 4096 8192; do
  for SEQ in 4096 16384 32768 65536; do
    OUT="scripts/_out/long_context/seq_${SEQ}_d${D}.json"
    if [[ -f "$OUT" ]]; then
      echo "[lc] SKIP existing: $OUT"
      continue
    fi
    # n_blocks * (seq_len * d * 2) bytes for activation ~= 8 * 65536 * 8192 * 2 = ~8 GB → fits.
    echo "===== seq=${SEQ} d=${D} ====="
    python scripts/micro_benchmark_primitives.py \
      --d "$D" --n_blocks 8 --seq_len "$SEQ" \
      --n_warmup 2 --n_iters 3 \
      --out "$OUT" 2>&1 | tail -12 | tee "${OUT%.json}.log.tail"
  done
done
echo "LONG_CONTEXT_SWEEP_DONE_$?"
