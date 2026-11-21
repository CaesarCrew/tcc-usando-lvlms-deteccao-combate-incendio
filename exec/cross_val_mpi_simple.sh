PROJECT_ROOT="$HOME/Development/tcc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

K=$(( $(nproc) / 2 ))
mpirun -np 2 --map-by slot:pe=$K --bind-to core -x OMP_NUM_THREADS=$K python3 -m training.cross_val_mpi \
--config "configs/CROSSVALMPI.yaml" --device "cpu" --llm_device "cpu" --k_folds 5 --epochs 1 --batch_size_train 1 --output_dir ./cv_outputs