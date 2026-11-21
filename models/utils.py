import os
import json
import torch

def load_sharded_bin_mmap(ckpt_dir):
    index_path = os.path.join(ckpt_dir, "pytorch_model.bin.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_files = ["pytorch_model.bin"]
 
    state_dict = {}
    for shard_file in shard_files:
        shard = torch.load(os.path.join(ckpt_dir, shard_file), map_location="cpu", mmap=True)
        state_dict.update(shard)
    return state_dict