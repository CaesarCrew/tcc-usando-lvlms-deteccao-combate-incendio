#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python3 -u training/cross_val.py \
--config "CROSSVAL.yaml" \
--device "cpu" \
--llm_device "cpu" \
--k_folds 10 \
--epochs 1 \
--batch_size_train 1 \
--output_dir ./cv_outputs