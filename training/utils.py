import json
import torch
import torch.nn as nn
import os

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
