#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python3 generate.py \
--config "configs/LYNX.yaml" \
--output_path "./stats.txt" \
--device "cuda" \
--k_folds 10
