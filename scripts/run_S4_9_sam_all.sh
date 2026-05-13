#!/usr/bin/env bash
# Training-free SAM RTN+DBAF for B/L/H on COCO val2017
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq:/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}

MAX=${1:-500}
for model_type in vit_b vit_l vit_h; do
  for flag in "" "--use-dbaf"; do
    suffix=baseline; [ -n "$flag" ] && suffix=with-dbaf
    name=sam-$(echo ${model_type} | tr _ -)
    out=results/S4-dbaf-weak/${name}-rtn/${suffix}/eval.json
    log=$out.log
    mkdir -p $(dirname $out)
    echo "===== ${name} ${suffix} (n=$MAX) ===== $(date)"
    python scripts/run_training_free_sam.py --model-type $model_type --bits 4 --max-images $MAX --out $out $flag 2>&1 | tee $log
  done
done
echo "S4_9_SAM_ALL_DONE_$?"
