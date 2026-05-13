#!/usr/bin/env bash
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ahcptq-old
export PYTHONPATH=/home/ubuntu/unifying-ptq:${PYTHONPATH:-}
mkdir -p results/S2-ahcptq/sam-b/smoke
timeout 1200 python ahcptq/solver/test_quant.py \
  --config ./projects/configs/yolox/yolo_l-sam-vit-b.py \
  --q_config ./exp/config44.yaml \
  --quant-encoder \
  --work-dir results/S2-ahcptq/sam-b/smoke \
  --eval segm 2>&1 | tee results/S2-ahcptq/sam-b/smoke/run.log
echo SMOKE_DONE_$?
