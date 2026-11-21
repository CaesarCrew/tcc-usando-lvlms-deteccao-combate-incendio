#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m training.cross_val \
--config "configs/CROSSVAL.yaml" \
--device "cpu" \
--llm_device "cpu" \
--k_folds 5 \
--epochs 1 \
--batch_size_train 1 \
--output_dir ./cv_outputs