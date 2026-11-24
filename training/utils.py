import json
import torch
import torch.nn as nn
import os
import gc
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

VISION_CONFIG_KEYS = ["image_res", "vision_encoder", "vit_ckpt_path", "use_left_pad"]


# Remove when push - prototype to shared memory model
class LynxFoldModel(nn.Module):
    """One of these per fold. Cheap: it only owns its own bridge.
    vision_encoder/LLM are references to the ONE shared, frozen model —
    not copies."""
    def __init__(self, shared_vision_encoder, shared_llm, shared_embed_tokens, bridge):
        super().__init__()
        self.vision_encoder = shared_vision_encoder  # shared reference
        self.LLM = shared_llm                        # shared reference
        self.embed_tokens = shared_embed_tokens       # shared reference
        self.bridge = bridge

def _load_mmap_state_dict(ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True)
    for n in list(state_dict.keys()):
        if '.attn.qkv.weight' in n:
            weight = state_dict[n]

            q, k, v = torch.split(weight, weight.shape[0]//3, dim=0)

            new_n = n.replace('.attn.qkv.weight', '.attn.q_proj.weight').strip()
            state_dict[new_n] = q

            new_n = n.replace('.attn.qkv.weight', '.attn.k_proj.weight').strip()
            state_dict[new_n] = k

            new_n = n.replace('.attn.qkv.weight', '.attn.v_proj.weight').strip()
            state_dict[new_n] = v

            del state_dict[n]

        elif '.attn.v_bias' in n:
            new_n = n.replace('.attn.v_bias', '.attn.v_proj.bias').strip()
            state_dict[new_n] = state_dict[n]
            del state_dict[n]

        elif '.attn.q_bias' in n:
            new_n = n.replace('.attn.q_bias', '.attn.q_proj.bias').strip()
            state_dict[new_n] = state_dict[n]
            del state_dict[n]
    
    return state_dict

def build_model_mmap(config, vision_device, llm_device):
    from models.lynx import LynxBase
 
    model = LynxBase(
        config=config,
        freeze_vit=config["freeze_vit"],
        freeze_llm=config["freeze_llm"],
        load_bridge=False,
    )

    if config["freeze_vit"]:
        vit_state = _load_mmap_state_dict("data/eva_vit_g.pth")
        model.vision_encoder.load_state_dict(vit_state, assign=True)
 
    model.vision_encoder = model.vision_encoder.to(vision_device)
    model.bridge = model.bridge.to(vision_device)
    #model.LLM = model.LLM.to(llm_device)
 
    if config["freeze_vit"]:
        model.vision_encoder = model.vision_encoder.half()
 
    return model

def build_model(config, vision_device, llm_device):
    from models.lynx import LynxBase

    model = LynxBase(
        config=config,
        freeze_vit=config["freeze_vit"],
        freeze_llm=config["freeze_llm"],
        load_bridge=False,
    )

    model.vision_encoder = model.vision_encoder.to(vision_device)
    model.bridge = model.bridge.to(vision_device)
    model.LLM = model.LLM.to(llm_device)

    if config["freeze_vit"]:
        model.vision_encoder = model.vision_encoder.half()

    return model


def save_fold_checkpoint(save_optimizer, model, rank, fold_id, epoch, checkpoint_dir, optimizer=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    cpath = os.path.join(checkpoint_dir, f"rank_{rank}_fold_{fold_id}_epoch_{epoch}.pt")
    if(save_optimizer):
        torch.save(
            {
                "fold": fold_id,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            cpath,
        )
    else:
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "args": vars(args)
            }, 
            cpath
        )

    print(f"### Fold {fold_id} checkpoint saved: {cpath}", flush=True)

def snapshot_trainable_state(model):
    """ Capture whatever the model's currently-trainable parameters hold
        right now. It goes by requires_grad, not by knowing module internals,
        so it's correct either way without needing that resolved first.
    """
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

def restore_trainable_state(model, snapshot):
    """ Reset every trainable parameter back to its snapshotted value, in
        place. Call this at the START of each new fold instead of rebuilding
        the model -- vision_encoder/LLM (frozen, shared/mmap'd per
        mmap_shared_loading.py) are never touched again after the first
        build, so the cross-rank sharing from that file stays intact, and
        this rank never re-parses the checkpoint files again either.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                p.data.copy_(snapshot[name])

def get_trainable_params(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    return trainable, n_params
    
def resolve_data_files(args, config):
    if "train_files" in config:
        return config["train_files"]
    return config["test_files"]


def vision_cache_key(config, data_files):
    files = [data_files] if isinstance(data_files, str) else sorted(data_files)
    ident = {
        "data_files": files,
        "num_frames": config.get("data", {}).get("num_frames"),
        **{k: config.get(k) for k in VISION_CONFIG_KEYS},
    }
    return hashlib.sha1(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _paths(cache_dir, key):
    base = os.path.join(cache_dir, f"vision_{key}")
    return base + ".npy", base + ".meta.json"


def vision_cache_exists(cache_dir, key):
    # The meta file is written LAST, so its presence means the cache is complete.
    return os.path.exists(_paths(cache_dir, key)[1])


@torch.no_grad()
def build_vision_cache(model, full_dataset, cache_dir, key, batch_size=16, num_workers=4):
    emb_path, meta_path = _paths(cache_dir, key)
    if os.path.exists(meta_path):
        print(f"### Vision cache already exists: {emb_path}", flush=True)
        return
    os.makedirs(cache_dir, exist_ok=True)

    if any(p.requires_grad for p in model.vision_encoder.parameters()):
        raise RuntimeError("Vision encoder is trainable (freeze_vit=False): its output "
                           "changes during training, so it can't be cached.")

    model.eval()
    vparam = next(model.vision_encoder.parameters())
    # shuffle=False + default sequential sampler -> batches come in dataset
    # order, so row i of the cache == full_dataset item i == the data_idx
    # LynxTrainDataset / Subset use.
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=full_dataset.collate_fn)

    n = len(full_dataset)
    tmp_path = emb_path + ".tmp"
    mm, pos, atts_template = None, 0, None

    for batch in loader:
        vision_input = batch[1].to(vparam.device, dtype=vparam.dtype, non_blocking=True)
        embeds, atts = model.get_vision_embeds(vision_input)

        if embeds.dtype == torch.float16:
            np_dtype = np.float16
        else:
            embeds = embeds.float()   # bf16 has no numpy dtype; fp32 is exact for it
            np_dtype = np.float32

        if mm is None:
            item_shape = tuple(embeds.shape[1:])
            if tuple(atts.shape) != tuple(embeds.shape[:-1]):
                raise RuntimeError(f"Unexpected vision_atts shape {tuple(atts.shape)} "
                                   f"for embeds {tuple(embeds.shape)}")
            atts_template = atts[0].detach().cpu()
            size_gb = n * int(np.prod(item_shape)) * np.dtype(np_dtype).itemsize / 1e9
            print(f"### Building vision cache: {n} items x {item_shape} {np_dtype.__name__} "
                  f"= {size_gb:.2f} GB on disk -> {emb_path}", flush=True)
            mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np_dtype, shape=(n,) + item_shape)

        # The cache stores ONE attention mask for every item. Refuse to build
        # if items actually differ (e.g. variable frame counts with padding).
        if not torch.equal(atts.detach().cpu(), atts_template.expand_as(atts.cpu())):
            raise RuntimeError("vision_atts differ between items; this cache stores a single "
                               "shared mask and can't represent that.")

        b = embeds.shape[0]
        mm[pos:pos + b] = embeds.cpu().numpy()
        pos += b
        if (pos // b) % 20 == 0 or pos == n:
            print(f"### vision cache: {pos}/{n}", flush=True)

    if pos != n:
        raise RuntimeError(f"Cached {pos} items but dataset has {n}")
    mm.flush()
    del mm
    os.replace(tmp_path, emb_path)

    meta = {
        "n": n,
        "item_shape": list(item_shape),
        "dtype": np.dtype(np_dtype).name,
        "atts_template": atts_template.tolist(),
        "atts_dtype": str(atts_template.dtype).replace("torch.", ""),
    }
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".tmp", meta_path)   # written last: marks the cache complete
    print(f"### Vision cache built: {emb_path}", flush=True)


class VisionCache:
    """ Read-only, memory-mapped view of the cached embeddings. Opened lazily
        in each process (including DataLoader workers), so every rank on a
        machine shares the same file pages instead of holding its own copy.
    """

    def __init__(self, cache_dir, key):
        self.emb_path, meta_path = _paths(cache_dir, key)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No complete vision cache at {self.emb_path}. "
                                    f"Build it first (single process).")
        with open(meta_path) as f:
            meta = json.load(f)
        self.n = meta["n"]
        self.atts_template = torch.tensor(meta["atts_template"],
                                          dtype=getattr(torch, meta["atts_dtype"]))
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            self._arr = np.load(self.emb_path, mmap_mode="r")
        return self._arr

    def __getstate__(self):           # don't pickle the open memmap into workers
        state = self.__dict__.copy()
        state["_arr"] = None
        return state

    def __len__(self):
        return self.n

    def get(self, i):
        return torch.from_numpy(np.array(self.arr[i]))   # copy one item out of the mmap

    def batch_atts(self, batch_size, device):
        shape = (batch_size,) + tuple(self.atts_template.shape)
        return self.atts_template.to(device).unsqueeze(0).expand(shape).clone()

# =========================================================================
# Model / optimizer helpers
# =========================================================================

def drop_vision_encoder(model):
    """ After the ViT is never used again once training and
        evaluation read the cache. Frees its GPU memory (and RAM).
    """
    model.vision_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_optimizer(params, lr, weight_decay, use_8bit=True):
    if use_8bit:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# =========================================================================
# Forward passes that start from cached embeddings
# =========================================================================

def bridge_from_cache(model, vision_embeds, vision_cache, llm_device, text_dtype):
    bparam = next(model.bridge.parameters())
    vision_embeds = vision_embeds.to(bparam.device, dtype=bparam.dtype, non_blocking=True)
    vision_atts = vision_cache.batch_atts(vision_embeds.shape[0], bparam.device)
    v2t_feats, v2t_atts = model.bridge(vision_embeds=vision_embeds, vision_atts=vision_atts)
    return v2t_feats.to(llm_device, dtype=text_dtype), v2t_atts.to(llm_device)

def setup_node_vision_cache(model, full_dataset, local_comm, is_worker,
                            cache_hosts, cache_dir, key, batch_size=8):
    """COLLECTIVE over local_comm: call it on EVERY rank of the job, the
    dispatcher included (pass model=None, is_worker=False there), at the
    same point -- after build_model() on workers, before the work-queue
    loop starts. Ranks that skip it would leave the others waiting.
 
    On a host listed in cache_hosts:
      - the first worker on that host builds the cache if it doesn't exist
        yet (using its own, already-built model's ViT);
      - the other workers on that host wait for it, then all of them open
        the cache and drop their ViT.
      Returns the VisionCache.
    On any other host (or on the dispatcher): returns None, and the worker
    keeps its ViT and the live (non-cached) path.
    """
    from mpi4py import MPI
 
    host = MPI.Get_processor_name()
    use_cache = is_worker and (host in cache_hosts or host.split(".")[0] in cache_hosts)
    print(f"### Host = {host}, use cache = {use_cache}", flush=True)
 
    # Sub-communicator of only the cache-using workers on this host.
    # Split is collective over local_comm, so every rank on the host calls it.
    cache_comm = local_comm.Split(color=0 if use_cache else MPI.UNDEFINED,
                                  key=local_comm.Get_rank())
    print(f"### rank {MPI.COMM_WORLD.Get_rank()} on '{host}': "
          f"vision cache {'ON' if use_cache else 'off'}", flush=True)
    if not use_cache:
        return None
 
    ok, err = True, None
    if cache_comm.Get_rank() == 0:
        try:
            build_vision_cache(model, full_dataset, cache_dir, key, batch_size=batch_size)
        except Exception as e:
            ok, err = False, repr(e)
    # bcast instead of a plain Barrier: if the build fails, the waiting
    # workers get the error and stop, instead of hanging forever.
    ok, err = cache_comm.bcast((ok, err), root=0)
    cache_comm.Free()
    if not ok:
        raise RuntimeError(f"vision cache build failed on {host}: {err}")
 
    cache = VisionCache(cache_dir, key)
    if len(cache) != len(full_dataset):
        raise RuntimeError(f"vision cache has {len(cache)} items, dataset has {len(full_dataset)}")
    
    if model.vision_encoder != None:
        drop_vision_encoder(model)
    return cache