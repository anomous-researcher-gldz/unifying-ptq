#!/usr/bin/env bash
# Control: FlatQuant + DBAF + PCSA WITHOUT KV-PCSA on LLaMA-3-8B (W4A4 KV4).
# Same recipe as run_S5_kv_pcsa_calib.sh but no --kv-pcsa flag.
set -e
cd /home/ubuntu/unifying-ptq/FlatQuant
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export HF_HOME=/data/huggingface_cache
export PYTHONPATH=/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}

OUT=/data/outputs/S5-baseline-calib
mkdir -p "$OUT/logs"

python main.py \
  --model /data/modelzoo/meta-llama/Meta-Llama-3-8B \
  --w_bits 4 --a_bits 4 \
  --k_bits 4 --k_asym --k_groupsize 128 \
  --v_bits 4 --v_asym --v_groupsize 128 \
  --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
  --lwc --lac --cali_trans --add_diag \
  --output_dir "$OUT" --save_matrix \
  --exp_name "fq-dbaf-pcsa-baseline" \
  2>&1 | tee "$OUT/logs/run.log"
echo "S5_BASELINE_CALIB_DONE_$?"
