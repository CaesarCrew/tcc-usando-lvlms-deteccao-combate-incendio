#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1

mpirun -np 16 python3 -u cross_val2_mpi.py \
--config "configs/CROSSVAL.yaml" \
--device "cuda" \
--llm_device "cpu" \
--k_folds 10 \
--epochs 100 \
--batch_size_train 1 \
--output_dir ./cv_outputs

echo $?