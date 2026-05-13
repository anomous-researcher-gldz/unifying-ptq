#!/usr/bin/env bash
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ahcptq-old
export PYTHONPATH=/home/ubuntu/unifying-ptq:${PYTHONPATH:-}
mkdir -p results/S2-ahcptq/sam-h/hdetr-w4a4/seed0/logs
python ahcptq/solver/test_quant.py \
  --config ./projects/configs/hdetr/r50-hdetr_sam-vit-h.py \
  --q_config ./exp/config44_samh.yaml \
  --quant-encoder \
  --work-dir results/S2-ahcptq/sam-h/hdetr-w4a4/seed0 \
  --save-pcsa results/S2-ahcptq/sam-h/hdetr-w4a4/seed0/pcsa.pt \
  --save_sam_path results/S2-ahcptq/sam-h/hdetr-w4a4/seed0/sam.pt \
  --eval segm 2>&1 | tee results/S2-ahcptq/sam-h/hdetr-w4a4/seed0/logs/run.log
echo SAMH_HDETR_DONE_$?
