#!/usr/bin/env bash
# F2: W4A4 dual-gate sweep on SwinIR-light x3 (Set5 first; quick).
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq

CKPT=/home/ubuntu/unifying-ptq/ckpt/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth
SET5=/home/ubuntu/unifying-ptq/data/sr_testsets/Set5_HR
URB=/home/ubuntu/unifying-ptq/data/sr_testsets/Urban100_HR
OUT=/home/ubuntu/unifying-ptq/results/F2-swinir-x3-dual-gate
mkdir -p "$OUT"

# All arms at W4A4. wg = weight gate, ag = activation gate.
# wg/ag values:
#   - 'force' (no gate, force-apply DBAF) -> --gate-frac3-max -1 (just for ablation; actually we'll omit the flag)
#   - 'default' -> 0.02
#   - 'off'    -> no DBAF on that side
# 6 arms: nodbaf | dbaf-Wforce-Aforce | dbaf-Wgate-Aforce | dbaf-Wforce-Agate | dbaf-Wgate-Agate | dbaf-Woff-Agate

run() {
  TAG=$1; shift
  for DS_PAIR in Set5:$SET5 Urban100:$URB; do
    DS_NAME=${DS_PAIR%:*}; DPATH=${DS_PAIR#*:}
    OUT_F="$OUT/${TAG}_${DS_NAME}.json"
    echo "=== $TAG / $DS_NAME ==="
    python scripts/run_training_free_swinir.py \
      --scale 3 --pretrained "$CKPT" --dataset "$DPATH" \
      --bits 4 --act-bits 4 \
      --out "$OUT_F" "$@"
  done
}

# Arm A: W4A4 RTN no DBAF
run "A_w4a4_nodbaf"

# Arm B: W4A4 + DBAF force on both (no gates anywhere)
run "B_w4a4_dbaf_Wforce_Aforce" --use-dbaf --act-gate-frac3-max -1

# Arm C: W4A4 + DBAF, default gate on weights, force on activations
run "C_w4a4_dbaf_Wgate_Aforce" --use-dbaf --gate-frac3-max 0.02 --act-gate-frac3-max -1

# Arm D: W4A4 + DBAF, force on weights, default gate on activations
run "D_w4a4_dbaf_Wforce_Agate" --use-dbaf --act-gate-frac3-max 0.02

# Arm E: W4A4 + DBAF, default gate on both (codebase-style)
run "E_w4a4_dbaf_Wgate_Agate" --use-dbaf --gate-frac3-max 0.02 --act-gate-frac3-max 0.02

# Arm F: W4A4 + DBAF strict-gate (0.01) on both
run "F_w4a4_dbaf_Wstrict_Astrict" --use-dbaf --gate-frac3-max 0.01 --act-gate-frac3-max 0.01

echo "F2_SWINIR_X3_DUAL_GATE_DONE_$?"
