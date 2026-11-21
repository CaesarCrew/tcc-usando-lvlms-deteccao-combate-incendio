#!/bin/bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="$HOME/Development/miniconda3/envs/lynx/bin/python" 
HOSTFILE="/home/natureman4everr/Development/tcc/hostfile.txt"
SCRIPT="training/cross_val_mpi.py"

LOCAL_IPS=$(ip -o addr show 2>/dev/null | awk '{split($4, a, "/"); print a[1]}' | tr '\n' ' ' || true)
NODE_NAME=$(uname -n 2>/dev/null || true)              # always present, unlike `hostname`
LOCAL_NAMES=" localhost 127.0.0.1 ::1 ${NODE_NAME} ${NODE_NAME%%.*} ${LOCAL_IPS} "
echo "### this machine: ${NODE_NAME:-?}, IPs: ${LOCAL_IPS:-none found}"
HOSTS=$(grep -v '^\s*#' "$HOSTFILE" | awk 'NF {print $1}')

COUNT_CORES="lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l"
N_NODES=$(echo "$HOSTS" | wc -w)
NP=$(( N_NODES * 2 ))
MIN_CORES=""
for h in $HOSTS; do
    if [[ "$LOCAL_NAMES" == *" $h "* ]]; then
        c=$(bash -c "$COUNT_CORES")
        where="local"
    else
        if ! c=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "$COUNT_CORES"); then
            echo "ERROR: cannot SSH to $h without a password -- mpirun needs that to start ranks there."
            echo "       Fix SSH first (see instructions), then test with:  ssh $h true"
            exit 1
        fi
        where="ssh"
    fi
    [[ "$c" =~ ^[0-9]+$ ]] || { echo "ERROR: couldn't read the core count on $h (got '$c')"; exit 1; }
    echo "### $h ($where): $c physical cores"
    if [ -z "$MIN_CORES" ] || [ "$c" -lt "$MIN_CORES" ]; then MIN_CORES=$c; fi
done
K=$(( MIN_CORES / 2 ))

cd "$HOME/Development/tcc"
mpirun --hostfile "$HOSTFILE" \
-np "$NP" \
--map-by "ppr:2:node:pe=${K}" \
 --bind-to core \
 -x OMP_NUM_THREADS=$K \
-x PATH -x LD_LIBRARY_PATH \
"$PYTHON" -u "$SCRIPT" "$@"
