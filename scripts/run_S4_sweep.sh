#!/usr/bin/env bash
# Sequential 6-cell sweep on one LLM: RTN/GPTQ/AWQ × {baseline, with-dbaf}
set -e
cd /home/ubuntu/unifying-ptq
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unifyptq
export PYTHONPATH=/home/ubuntu/unifying-ptq/FlatQuant:${PYTHONPATH:-}
export HF_HOME=/data/huggingface_cache

MODEL=${1:-llama3-8b}
mkdir -p results/S4-dbaf-weak/${MODEL}

for baseline in rtn awq gptq; do  # fastest first
  for flag in "" "--use-dbaf"; do
    suffix=baseline; [ -n "$flag" ] && suffix=with-dbaf
    out=results/S4-dbaf-weak/${MODEL}/${baseline}/${suffix}/eval.json
    log=results/S4-dbaf-weak/${MODEL}/${baseline}/${suffix}/run.log
    mkdir -p "$(dirname "$out")"
    echo "===== ${baseline} ${suffix} ===== $(date)" >> "$log"
    python scripts/run_S4.py --model "$MODEL" --baseline "$baseline" $flag --bits 4 --out "$out" 2>&1 | tee -a "$log"
  done
done
echo "S4_SWEEP_${MODEL}_DONE_$?"
