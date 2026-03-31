#!/bin/bash

# Activate Conda
# source /opt/conda/etc/profile.d/conda.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate srtquant

printf -v joined '%s_' "$@"     # e.g., "foo_bar_baz_"
joined=${joined%_} 

export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:21
# Run Python
exec python basicsr/train.py -opt options/train/train_srtquant_x4.yml "$@" \
       --force_yml bit=4 name="srtquant_x4_bit4_${joined}"
       
# bits=(3 4)
# prunes=(0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99 0.995)

# for bit in "${bits[@]}"; do
#   for pct in "${prunes[@]}"; do                            
#     run_name="srtquant_x4_bit${bit}_prune${pct_tag}${suffix}"
#     echo "=== Running ${run_name} (bit=${bit}, prune_percent=${pct}) ==="

#     python basicsr/train.py -opt options/train/train_srtquant_x4.yml "$@" \
#       --force_yml \
#         bit="${bit}" \
#         prune_percent="${pct}" \
#         name="${run_name}"
#   done
# done