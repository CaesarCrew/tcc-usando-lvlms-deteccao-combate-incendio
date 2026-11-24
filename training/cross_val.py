import argparse
import gc
import resource
import copy
import datetime as dt
from datetime import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset, Subset

from calculate_metrics import process_metrics
from dataset import create_dataset
from utils import get_rank, write_jsonl


CROSSVAL_LYNX_INTEGRATION_VERSION = "crossval-lowram-reuse-safe-v18-2026-11-24"


def cuda_memory_report(tag):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    max_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(
        f"### CUDA memory [{tag}]: allocated={allocated:.1f} MiB, reserved={reserved:.1f} MiB, "
        f"max_allocated={max_allocated:.1f} MiB, max_reserved={max_reserved:.1f} MiB",
        flush=True,
    )


def ram_memory_report(tag):
    """Log physical RSS and cgroup usage without extra dependencies.

    Linux cgroup memory.current also includes file cache, while /proc/self/
    status VmRSS reflects resident process pages (including mapped files).
    Neither number is equivalent to the sum of Python tensor sizes.
    """
    rss = None
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    # ru_maxrss is KiB on Linux (Google Colab); tracks transient peak RSS.
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except (OSError, ValueError):
        peak_rss = None
    cgroup = None
    for filename in ('/sys/fs/cgroup/memory.current',
                     '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            with open(filename, 'r', encoding='utf-8') as handle:
                cgroup = int(handle.read().strip())
            break
        except (OSError, ValueError):
            continue
    def format_gib(value):
        return 'n/a' if value is None else f'{value / 1024**3:.2f} GiB'
    print(f'### RAM [{tag}]: process RSS={format_gib(rss)}, '
          f'peak RSS={format_gib(peak_rss)}, '
          f'cgroup(includes file cache)={format_gib(cgroup)}', flush=True)


class TrainFoldDataset(Dataset):
    def __init__(self, base_dataset, indices, max_total_tokens=128,
                 image_res=224, image_mean=None, image_std=None):
        self.base_dataset = base_dataset
        self.image_res = int(image_res)
        self.image_mean = image_mean or [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std or [0.26862954, 0.26130258, 0.27577711]
        self.indices = list(indices)
        self.max_total_tokens = max_total_tokens

        self.tokenizer = base_dataset.tokenizer
        self.use_left_pad = base_dataset.use_left_pad
        self.lower_text = base_dataset.lower_text
        self.output_key = base_dataset.OUTPUT_PROMPT_DICT

    def __len__(self):
        return len(self.indices)

    def _truncate(self, prompt_ids, answer_ids):
        max_total = max(2, self.max_total_tokens)

        if len(answer_ids) >= max_total:
            answer_ids = answer_ids[-(max_total - 1):]
            prompt_ids = prompt_ids[:1]

        total = len(prompt_ids) + len(answer_ids)
        if total <= max_total:
            return prompt_ids, answer_ids

        keep_prompt = max(1, max_total - len(answer_ids))
        if keep_prompt == 1:
            return prompt_ids[:1], answer_ids

        return [prompt_ids[0]] + prompt_ids[-(keep_prompt - 1):], answer_ids

    def __getitem__(self, item):
        idx = self.indices[item]
        ann = self.base_dataset.data[idx]

        if self.output_key not in ann:
            raise KeyError(f"Missing key '{self.output_key}' in sample index {idx}")

        vision_input = self.base_dataset.get_vision_input(ann)
        prompt_text = self.base_dataset.get_text_input(ann)

        answer_text = str(ann[self.output_key]).strip()
        if self.lower_text:
            answer_text = answer_text.lower()

        prompt_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(prompt_text)
        answer_tokens = self.tokenizer.tokenize(" " + answer_text) + [self.tokenizer.eos_token]

        prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)

        prompt_ids, answer_ids = self._truncate(prompt_ids, answer_ids)

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids

        return idx, vision_input, input_ids, labels

    def _pad_sequences(self, input_ids, labels):
        max_len = max(len(x) for x in input_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids_pad = []
        input_atts_pad = []
        labels_pad = []

        for ids, lbs in zip(input_ids, labels):
            n = len(ids)
            n_pad = max_len - n

            if self.use_left_pad:
                input_ids_pad.append(([pad_id] * n_pad) + ids)
                input_atts_pad.append(([0] * n_pad) + ([1] * n))
                labels_pad.append(([-100] * n_pad) + lbs)
            else:
                input_ids_pad.append(ids + ([pad_id] * n_pad))
                input_atts_pad.append(([1] * n) + ([0] * n_pad))
                labels_pad.append(lbs + ([-100] * n_pad))

        return (
            torch.LongTensor(input_ids_pad),
            torch.LongTensor(input_atts_pad),
            torch.LongTensor(labels_pad)
        )

    def collate_fn(self, batch):
        """Collate vision tensors; normalize PIL frames exactly once.

        The project dataset normally returns already-preprocessed tensors,
        which MUST NOT be normalized twice. A raw PIL fallback is explicitly
        normalized using CROSSVAL.yaml's image_mean/image_std instead of the
        old ToTensor-only fallback. Invalid samples raise rather than being
        silently replaced by gray images with potentially wrong labels.
        """
        idx, vision_samples, input_ids, labels = zip(*batch)
        from PIL import Image
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((self.image_res, self.image_res)),
            T.ToTensor(),
            T.Normalize(self.image_mean, self.image_std),
        ])

        def as_tensor(image):
            if isinstance(image, torch.Tensor):
                return image
            if isinstance(image, Image.Image):
                return transform(image.convert('RGB'))
            raise TypeError(f'Unsupported training image type: {type(image).__name__}')

        converted = []
        for sample in vision_samples:
            if isinstance(sample, (tuple, list)):
                converted.append(torch.stack([as_tensor(frame) for frame in sample], dim=0))
            else:
                converted.append(as_tensor(sample))
        vision_input = torch.stack(converted, dim=0)
        # Tensor shape is [B,C,H,W] for single images or [B,F,C,H,W]
        # for video; no flatten/reshape with potentially incorrect frame counts.
        input_ids_pad, input_atts_pad, labels_pad = self._pad_sequences(input_ids, labels)
        return idx, vision_input, input_ids_pad, input_atts_pad, labels_pad

def make_kfold_indices(annotations, k=5, seed=123, stratified=True):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(annotations))

    if not stratified:
        rng.shuffle(all_indices)
        return [arr.tolist() for arr in np.array_split(all_indices, k)]

    label_to_indices = {}
    for i, ann in enumerate(annotations):
        label = str(ann.get("answer", "")).strip().lower()
        label_to_indices.setdefault(label, []).append(i)

    folds = [[] for _ in range(k)]
    for _, label_indices in sorted(label_to_indices.items(), key=lambda kv: kv[0]):
        label_indices = np.array(label_indices)
        rng.shuffle(label_indices)
        for j, idx in enumerate(label_indices.tolist()):
            folds[j % k].append(idx)

    for fold in folds:
        fold.sort()

    return folds


def _resolve_requested_device(value, role):
    """Accept --device/--llm_device cpu, gpu, cuda, or cuda:N.

    Resolve the requested device once; LynxBase may offload components later.
    """
    if isinstance(value, torch.device):
        target = value
    else:
        raw = str(value).strip().lower()
        if raw == "gpu":
            raw = "cuda"
        try:
            target = torch.device(raw)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid {role}={value!r}; choose cpu, gpu, cuda or cuda:N") from exc
    if target.type not in ("cpu", "cuda"):
        raise ValueError(f"Invalid {role}: {target}; only CPU and CUDA are supported")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{role}={value!s} requests CUDA, but this runtime has no CUDA GPU")
        if target.index is not None and target.index >= torch.cuda.device_count():
            raise ValueError(f"{role} CUDA index {target.index} is unavailable")
    return target


def _module_device_dtype(module):
    """Inspect the actual placement selected by the Lynx VRAM planner."""
    param = next(module.parameters(), None)
    if param is not None:
        return param.device, param.dtype
    buf = next(module.buffers(), None)
    return (buf.device, buf.dtype) if buf is not None else (torch.device("cpu"), torch.float32)


def build_model(config, device, llm_device="cpu", train_adapters=False):
    from models import lynx as lynx_module
    LynxBase = lynx_module.LynxBase
    required_version = 'sequential-zero-copy-activation-checkpoint-v18-2026-11-24'
    loaded_version = getattr(lynx_module, 'LYNX_TRAIN_NUMERICS_PATCH_VERSION', None)
    if loaded_version != required_version:
        raise RuntimeError(
            f'cross_val.py V18 requires models/lynx.py V18, got {loaded_version!r} '
            f'from {getattr(lynx_module, "__file__", "<unknown>")}. '
            'Replace BOTH files and restart the Colab runtime.'
        )
    print(f'### Crossval code: {CROSSVAL_LYNX_INTEGRATION_VERSION}', flush=True)
    print(f'### Lynx code: {loaded_version}', flush=True)

    requested_vision = _resolve_requested_device(device, "device")
    requested_llm = _resolve_requested_device(llm_device, "llm_device")

    if train_adapters and not config.get("use_adapter", False):
        raise ValueError(
            "--train_adapters requires use_adapter: true in CROSSVAL.yaml. "
            "Also match adapter_freq to the checkpoint's training configuration."
        )
    if config.get('use_adapter', False):
        frequency = config.get('adapter_freq', None)
        if not isinstance(frequency, int) or isinstance(frequency, bool) or frequency <= 0:
            raise ValueError('use_adapter: true requires adapter_freq to be a positive integer')
        print(f'### Crossval: constructing an output adapter every {frequency} '
              'LLaMA layers; strict checkpoint verification is '
              f'{bool(config.get("strict_checkpoint_adapters", False))}', flush=True)

    # V9 numerical-safety policy: the frozen EVA-ViT is run in FP32 by
    # default during training. A float16 ViT can overflow internally; a FP32
    # bridge cannot undo NaNs that its encoder has already produced.
    # The Lynx VRAM planner sees these choices *before* moving any weights.
    effective_config = copy.deepcopy(config)
    if not effective_config.get('freeze_vit', True):
        # Preserve the constructor's original flag because the EVA-ViT
        # pretrained loader may report missing keys that the *full* Lynx
        # checkpoint subsequently supplies. The CV optimizer still freezes
        # the vision encoder explicitly AFTER all weights are loaded.
        print('### Crossval: constructing ViT with freeze_vit=False for '
              'pretrained-loader compatibility; freezing it before training',
              flush=True)
    vision_precision = effective_config.get("crossval_vision_precision", "float32")
    bridge_precision = effective_config.get("crossval_bridge_precision", "float32")
    allowed = ("float16", "bfloat16", "float32", "auto")
    if vision_precision not in allowed or bridge_precision not in allowed:
        raise ValueError("crossval_vision_precision and crossval_bridge_precision "
                         "must be float16, bfloat16, float32, or auto")
    if requested_vision.type == "cuda":
        if vision_precision != "auto":
            effective_config["gpu_vision_dtype"] = vision_precision
        if bridge_precision != "auto":
            effective_config["gpu_bridge_dtype"] = bridge_precision
    print(f"### Crossval stable training precision: "
          f"vision={effective_config.get('gpu_vision_dtype', 'float32')}, "
          f"bridge={effective_config.get('gpu_bridge_dtype', 'float32')}", flush=True)

    if requested_vision.type == "cuda" or requested_llm.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        cuda_memory_report("before LynxBase ctor")

    model = LynxBase(
        config=effective_config,
        freeze_vit=effective_config["freeze_vit"],
        freeze_llm=effective_config["freeze_llm"],
        load_bridge=False,
        device=requested_vision,
        llm_device=requested_llm,
    )
    # Important: do NOT call .to(), .half(), or .cuda() here. LynxBase has
    # already chosen placements based on free VRAM, including CPU offload.
    cuda_memory_report("after LynxBase ctor / automatic placement")

    vision_actual, vision_dtype = _module_device_dtype(model.vision_encoder)
    bridge_actual, bridge_dtype = _module_device_dtype(model.bridge)
    llm_actual, llm_dtype = _module_device_dtype(model.LLM)
    print(
        "### Crossval actual placement: "
        f"vision={vision_actual} ({vision_dtype}), "
        f"bridge={bridge_actual} ({bridge_dtype}), "
        f"LLM={llm_actual} ({llm_dtype})",
        flush=True,
    )
    if llm_actual.type == "cpu" and requested_llm.type == "cuda":
        print("### Crossval: Lynx offloaded LLM to CPU to respect VRAM budget", flush=True)

    # Confirm checkpoint adapter weights were actually restored before
    # optimizing or reporting fold metrics. An unloaded adapter checkpoint
    # silently changes the model under evaluation.
    coverage = getattr(model, '_lynx_checkpoint_load_report', None)
    if coverage is not None and effective_config.get('strict_checkpoint_adapters', False):
        print(f"### Crossval checkpoint adapter coverage: "
              f"{coverage['adapter_loaded']}/{coverage['adapter_expected']}", flush=True)

    # Freeze pretrained weights, then enable only intended training modules.
    for p in model.parameters():
        p.requires_grad_(False)
    # Make the actual forward path run without a ViT autograd graph, even if
    # the loader required freeze_vit=False during checkpoint construction.
    model.freeze_vit = True
    model.vision_encoder.eval()
    for p in model.bridge.parameters():
        p.requires_grad_(True)

    if train_adapters:
        adapter_count = 0
        for name, param in model.LLM.named_parameters():
            if "adapter" in name:
                param.requires_grad_(True)
                adapter_count += 1
        if adapter_count == 0:
            raise RuntimeError(
                "--train_adapters requested, but LLM contains no adapter parameters. "
                "Check use_adapter and adapter_freq."
            )
        print(f"### Crossval training {adapter_count} adapter tensors", flush=True)

    # The decoder still explicitly enables KV caching for generation. During
    # cross-validation training KV caching is unnecessary and wastes memory.
    if hasattr(model, "prepare_crossval_training"):
        model.prepare_crossval_training()
    model._crossval_loss_mode = effective_config.get("crossval_loss_mode", "stable")
    if model._crossval_loss_mode not in ("stable", "legacy"):
        raise ValueError("crossval_loss_mode must be stable or legacy")
    model._crossval_loss_label_smoothing = float(
        effective_config.get("label_smoothing", 0.0))
    if not 0.0 <= model._crossval_loss_label_smoothing < 1.0:
        raise ValueError("label_smoothing must be >= 0 and < 1")

    model._crossval_only_supervised_logits = bool(
        effective_config.get('crossval_only_supervised_logits', True))
    model._crossval_llm_trace_on_error = bool(
        effective_config.get('crossval_llm_trace_on_error', True))
    print(f'### Crossval numerics: supervised-only LM head='
          f'{model._crossval_only_supervised_logits}, '
          f'trace_on_error={model._crossval_llm_trace_on_error}', flush=True)
    cuda_memory_report("after model parameter freezing")
    return model

class TrainingNumericalError(RuntimeError):
    """Training input or model activation contains non-finite values."""


def _require_finite(tensor, name, indices=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(tensor).__name__}")
    valid = torch.isfinite(tensor)
    if bool(valid.all()):
        return
    bad_count = int((~valid).sum().item())
    finite = tensor.detach()[valid]
    value_range = (f"finite_range=[{float(finite.min()):.5g}, "
                   f"{float(finite.max()):.5g}]" if finite.numel() else "NO finite values")
    samples = f"; sample_ids={list(indices)}" if indices is not None else ""
    raise TrainingNumericalError(
        f"stage={name}: {bad_count}/{tensor.numel()} non-finite values, "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, {value_range}{samples}. "
        "Do not replace NaNs with zeros: inspect image preprocessing, "
        "checkpoint weights, and precision."
    )


def _stable_next_token_ce(logits, labels, *, smoothing=0.0, indices=None):
    """Calculate causal LM loss in float32 only on supervised target tokens.

    This avoids FP16 overflow in the legacy LLaMA loss code and does not
    produce NaN from a mean over zero non-ignored labels. It preserves the
    standard one-token shift and ignore_index=-100 semantics.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(f"Invalid logits/labels shapes: "
                         f"{tuple(logits.shape)} / {tuple(labels.shape)}")
    if logits.shape[1] < 2:
        raise ValueError("Causal language-model training requires at least 2 tokens")
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels.ne(-100)
    count = int(supervised.sum().item())
    if not count:
        raise TrainingNumericalError(
            f"stage=targets: zero supervised tokens after causal shift "
            f"(labels shape={tuple(labels.shape)}, sample_ids={list(indices)}). "
            "Check target generation and truncation; loss would be NaN."
        )
    targets = shifted_labels[supervised]
    vocab = logits.shape[-1]
    if bool((targets < 0).any()) or bool((targets >= vocab).any()):
        raise ValueError(f"Targets outside vocabulary range [0, {vocab}): "
                         f"min={int(targets.min())}, max={int(targets.max())}")

    # Select before converting to FP32: no need to duplicate *every* vocabulary
    # logit at ignored image/prompt positions; gradients still flow normally.
    chosen_logits = logits[:, :-1, :][supervised]
    _require_finite(chosen_logits, "llm_supervised_logits", indices)
    loss = torch.nn.functional.cross_entropy(
        chosen_logits.float(), targets.long(),
        reduction="mean", label_smoothing=smoothing,
    )
    _require_finite(loss, "stable_float32_cross_entropy", indices)
    return loss


def _first_tensor_in_output(output):
    """Extract a model output tensor without retaining intermediate activations."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _first_tensor_in_output(output[0])
    for attr in ('last_hidden_state', 'logits'):
        item = getattr(output, attr, None)
        if isinstance(item, torch.Tensor):
            return item
    return None


def _trace_llm_first_nonfinite(model, inputs_embeds, attention_mask,
                               position_ids, indices):
    """Run ONE diagnostic forward after failure; stop at the first bad module.

    Hooks are installed only for the error replay. No parameters or original
    forward functions are changed; the replay does not update any weights.
    """
    llm = model.LLM
    root = getattr(llm, 'model', None)
    if root is None:
        return 'LLM has no .model; cannot trace'
    handles = []
    def inspect_output(name):
        def hook(_module, _inputs, output):
            value = _first_tensor_in_output(output)
            if value is not None:
                try:
                    _require_finite(value.detach(), f'llm.{name}', indices)
                except TrainingNumericalError as exc:
                    if value.ndim >= 3:
                        bad = (~torch.isfinite(value.detach())).any(dim=-1)
                        rows = torch.nonzero(bad, as_tuple=False)[:12].cpu().tolist()
                        raise TrainingNumericalError(
                            f'{exc}; bad_batch_token_positions={rows}; '
                            f'backend={getattr(getattr(llm.model.layers[int(name.split(".")[1])], "self_attn", None), "_lynx_stable_sdpa", None) if name.startswith("layers.") else "n/a"}') from exc
                    raise
        return hook

    def inspect_input(name):
        def hook(_module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                value = inputs[0]
                if not bool(torch.isfinite(value.detach()).all()):
                    bad_rows = torch.nonzero(
                        (~torch.isfinite(value.detach())).any(dim=-1),
                        as_tuple=False)[:12].cpu().tolist()
                    raise TrainingNumericalError(
                        f'FIRST_BAD_ATTENTION_INPUT stage=llm.{name}: '
                        f'nonfinite token positions={bad_rows}; sample_ids={list(indices)}; '
                        'attention scores or values became non-finite before '
                        'the output projection. Examine q/k/v ranges and mask.')
        return hook

    try:
        layers = getattr(root, 'layers', ())
        for i, layer in enumerate(layers):
            # Inspect the deepest components BEFORE the enclosing layer.
            for name in ('input_layernorm', 'self_attn', 'post_attention_layernorm',
                         'mlp', 'output_adapter'):
                child = getattr(layer, name, None)
                if child is None:
                    continue
                if name == 'self_attn':
                    for sub in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                        node = getattr(child, sub, None)
                        if node is not None:
                            if sub == 'o_proj':
                                handles.append(node.register_forward_pre_hook(
                                    inspect_input(f'layers.{i}.self_attn.before_o_proj')))
                            handles.append(node.register_forward_hook(
                                inspect_output(f'layers.{i}.self_attn.{sub}')))
                elif name == 'mlp':
                    for sub in ('gate_proj', 'up_proj', 'down_proj'):
                        node = getattr(child, sub, None)
                        if node is not None:
                            handles.append(node.register_forward_hook(
                                inspect_output(f'layers.{i}.mlp.{sub}')))
                handles.append(child.register_forward_hook(
                    inspect_output(f'layers.{i}.{name}')))
            handles.append(layer.register_forward_hook(
                inspect_output(f'layers.{i}.output')))
        norm = getattr(root, 'norm', None)
        if norm is not None:
            handles.append(norm.register_forward_hook(
                inspect_output('final_norm')))
        with torch.no_grad():
            output = root(
                inputs_embeds=inputs_embeds.detach(),
                attention_mask=attention_mask.detach(),
                position_ids=position_ids.detach(),
                use_cache=False, return_dict=True,
            )
            hidden = _first_tensor_in_output(output)
            if hidden is not None:
                _require_finite(hidden, 'llm.final_hidden', indices)
    except TrainingNumericalError as exc:
        return str(exc)
    except Exception as exc:
        return f'diagnostic replay failed with {type(exc).__name__}: {exc}'
    finally:
        for handle in handles:
            handle.remove()
    return ('all traced LLM hidden states finite; inspect lm_head weights, '
            'supervised token projection, and the training-time execution mode')


def _supervised_token_ce(model, hidden, labels, *, smoothing, indices):
    """Compute causal CE using only the positions with supervised targets.

    Avoids a [batch, full-sequence, vocab] allocation; this is mathematically
    equivalent to the usual language-model head when it acts per token.
    """
    if hidden.ndim != 3 or labels.ndim != 2 or hidden.shape[:2] != labels.shape:
        raise ValueError(f'hidden/labels shape mismatch: '
                         f'{tuple(hidden.shape)} versus {tuple(labels.shape)}')
    supervised = labels[:, 1:].ne(-100)
    if not bool(supervised.any()):
        raise TrainingNumericalError('stage=targets: no supervised next-token targets')
    targets = labels[:, 1:][supervised]
    head = model.LLM.get_output_embeddings()
    if head is None:
        raise RuntimeError('LLM output embedding head unavailable')
    selected = hidden[:, :-1, :][supervised]
    _require_finite(selected, 'llm_supervised_hidden_states', indices)
    # Keep head computation in native dtype (CPU FP32 in offload mode), but
    # compute cross-entropy in FP32 for stable normalization.
    head_weight = next(head.parameters(), None)
    if head_weight is not None and head_weight.dtype != selected.dtype:
        selected = selected.to(head_weight.dtype)
    logits = head(selected)
    _require_finite(logits, 'llm_supervised_logits', indices)
    vocab = logits.size(-1)
    if bool((targets < 0).any()) or bool((targets >= vocab).any()):
        raise ValueError(f'Supervised target outside [0, {vocab})')
    loss = torch.nn.functional.cross_entropy(
        logits.float(), targets.long(), label_smoothing=float(smoothing))
    _require_finite(loss, 'supervised_only_ce', indices)
    return loss


def forward_train_loss(model, batch, device):
    indices, vision_input, input_ids, input_atts, labels = batch

    # Use *actual* device selected by Lynx's VRAM offloader (not the CLI flag).
    llm_device = model.LLM.model.embed_tokens.weight.device
    input_ids = input_ids.to(llm_device, non_blocking=True)
    input_atts = input_atts.to(llm_device, non_blocking=True)
    labels = labels.to(llm_device, non_blocking=True)
    text_embeds = model.embed_tokens(input_ids)
    _require_finite(text_embeds, "text_embeddings", indices)

    if vision_input is not None:
        _require_finite(vision_input, "input_images", indices)
        with torch.set_grad_enabled(not model.freeze_vit):
            vision_embeds, vision_atts = model.get_vision_embeds(vision_input)
        _require_finite(vision_embeds, "vision_encoder_output", indices)

        bridge_device, bridge_dtype = _module_device_dtype(model.bridge)
        vision_embeds = vision_embeds.to(
            device=bridge_device, dtype=bridge_dtype, non_blocking=True
        )
        vision_atts = vision_atts.to(bridge_device, non_blocking=True)
        v2t_feats, v2t_atts = model.bridge(
            vision_embeds=vision_embeds, vision_atts=vision_atts
        )
        _require_finite(v2t_feats, "bridge_output", indices)
        v2t_feats = v2t_feats.to(
            device=llm_device, dtype=text_embeds.dtype, non_blocking=True
        )
        # The fixed-query Perceiver has no padding: never use non-binary
        # bridge-returned values as an attention mask.
        from models.lynx import _lynx_resampler_attention_mask
        v2t_atts = _lynx_resampler_attention_mask(
            v2t_feats, v2t_atts, name='crossval_bridge_mask')
        ignore_labels = torch.full(
            (labels.size(0), v2t_feats.size(1)), -100,
            dtype=labels.dtype, device=llm_device,
        )
        inputs_embeds = torch.cat((v2t_feats, text_embeds), dim=1)
        attention_mask = torch.cat((v2t_atts, input_atts), dim=1)
        labels = torch.cat((ignore_labels, labels), dim=1)
    else:
        inputs_embeds, attention_mask = text_embeds, input_atts

    # Check masks and positions before reaching legacy LLaMA RoPE.
    if inputs_embeds.shape[:2] != labels.shape or labels.shape != attention_mask.shape:
        raise ValueError("Multimodal training mask/label length mismatch")
    _require_finite(inputs_embeds, "llm_inputs_embeds", indices)
    if not getattr(model, '_crossval_logged_input_scale', False):
        bridge_max = float(v2t_feats.detach().abs().amax()) if vision_input is not None else 0.0
        text_max = float(text_embeds.detach().abs().amax())
        print(f'### Crossval LLM input magnitudes: '
              f'max_abs_bridge={bridge_max:.6g}, max_abs_text={text_max:.6g}; '
              'V17 checks RMSNorm, projections and RoPE separately', flush=True)
        model._crossval_logged_input_scale = True
    # Use the same bounded, binary-mask RoPE positions as generation.
    # Legacy LLaMA constructs its rotary table for exactly the current length.
    from models.lynx import _lynx_normalize_attention_mask, _lynx_positions_from_mask
    attention_mask = _lynx_normalize_attention_mask(
        attention_mask, name='crossval_train_attention_mask')
    position_ids = _lynx_positions_from_mask(
        attention_mask, current_length=inputs_embeds.size(1),
        name='crossval_train_attention_mask')

    mode = getattr(model, "_crossval_loss_mode", "stable")
    if mode == "legacy":
        outputs = model.LLM(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, labels=labels, use_cache=False,
            return_dict=True,
        )
        _require_finite(outputs.loss, "legacy_llm_loss", indices)
        return outputs.loss

    try:
        if getattr(model, '_crossval_only_supervised_logits', True):
            # Directly run LLaMA's transformer, then apply its unchanged
            # output head only to supervised positions. This both decreases
            # memory and isolates the source of non-finite activations.
            lm_result = model.LLM.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False, return_dict=True,
            )
            hidden = _first_tensor_in_output(lm_result)
            if hidden is None:
                raise RuntimeError('LLaMA backbone returned no hidden states')
            return _supervised_token_ce(
                model, hidden, labels,
                smoothing=getattr(model, '_crossval_loss_label_smoothing', 0.0),
                indices=indices,
            )
        outputs = model.LLM(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, labels=None, use_cache=False,
            return_dict=True,
        )
        if getattr(outputs, 'logits', None) is None:
            raise RuntimeError('LLaMA did not return logits')
        return _stable_next_token_ce(
            outputs.logits, labels,
            smoothing=getattr(model, '_crossval_loss_label_smoothing', 0.0),
            indices=indices,
        )
    except TrainingNumericalError as exc:
        if not getattr(model, '_crossval_llm_trace_on_error', True):
            raise
        cause = _trace_llm_first_nonfinite(
            model, inputs_embeds, attention_mask, position_ids, indices)
        raise TrainingNumericalError(f'{exc}; FIRST_BAD_LLM_STAGE: {cause}') from exc
    except FloatingPointError as exc:
        # V17's attention/RMSNorm pre-check identifies the *true first* invalid
        # calculation. Do not replace NaNs with zeros or retry a bad batch.
        raise TrainingNumericalError(
            f'FIRST_BAD_LLM_STAGE: {exc}; sample_ids={list(indices)}') from exc


def train_one_fold(model, train_loader, device, epochs, lr, weight_decay, grad_accum_steps, grad_clip):
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    model.train()
    model.vision_encoder.eval()
    if not any(p.requires_grad for p in model.LLM.parameters()):
        model.LLM.eval()  # Frozen LM: deterministic dropout while preserving input gradients

    # Examine only trainable tensors (normally the much smaller bridge), not
    # every parameter of the 7B frozen LLM. Fail before the first optimizer step.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            _require_finite(parameter.detach(), f'trainable_weight:{name}')

    history = []
    global_step = 0

    for epoch in range(epochs):
        running_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if epoch == 0 and step == 0:
                first_images = batch[1]
                print(f'### Crossval first training batch: '
                      f'images_shape={tuple(first_images.shape)}, '
                      f'images_dtype={first_images.dtype}, '
                      f'images_range=({float(first_images.min()):.5g}, '
                      f'{float(first_images.max()):.5g}); '
                      f'labels_supervised={int(batch[4].ne(-100).sum())}', flush=True)
            try:
                loss = forward_train_loss(model, batch, device)
            except TrainingNumericalError as exc:
                raise RuntimeError(
                    f"Training numerical failure at epoch={epoch + 1}, "
                    f"batch={step + 1}, {exc}. "
                    'Verify checkpoint adapter coverage, the model version, '
                    'and the stage recorded above before changing precision.'
                ) from exc
            _require_finite(loss, "train_loss", batch[0])
            loss = loss / grad_accum_steps
            loss.backward()
            print(f"### Step {step + 1}/{len(train_loader)}")
            if (step + 1) % grad_accum_steps == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip, error_if_nonfinite=True)
                for param in trainable_params:
                    if param.grad is not None:
                        _require_finite(param.grad, "trainable_gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss.item() * grad_accum_steps
            n_steps += 1

        if n_steps % grad_accum_steps != 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip, error_if_nonfinite=True)
            for param in trainable_params:
                if param.grad is not None:
                    _require_finite(param.grad, "trainable_gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        epoch_loss = running_loss / max(1, n_steps)
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "global_step": global_step})

    return history


def _process_metrics_with_empty_guard(annotations, predictions):
    """Preserve raw predictions, protect the existing metrics parser.

    calculate_metrics.predict_binary_class currently accesses
    `lead_text.split(maxsplit=1)[0]` even if cleaning leaves no tokens.
    Rather than rewriting the user's separate metrics module, probe the
    parser with each prediction first. ONLY on that specific IndexError,
    present the existing parser with an empty string, following the code's
    already-defined empty-input branch. This does not invent a class or
    substitute an artificial Fire/No-Fire response.

    All raw model responses remain unchanged in prediction JSONL. Reports
    additionally record counts/IDs of responses needing normalization.
    """
    from calculate_metrics import predict_binary_class
    metric_predictions = []
    unparseable = []
    blank = []
    for record in predictions:
        text = record['text_output']
        if not isinstance(text, str):
            raise TypeError(f'Prediction {record["index"]} is not text')
        if not text.strip():
            blank.append(record['index'])
        try:
            predict_binary_class(text)
            metric_predictions.append(record)
        except IndexError:
            # An IndexError is known to arise from `lead_text.split()[0]`.
            # Verify that calculate_metrics can parse its *own* empty branch.
            predict_binary_class('')
            fixed = dict(record)
            fixed['text_output'] = ''
            metric_predictions.append(fixed)
            unparseable.append(record['index'])
    if unparseable:
        print('### Crossval warning: metrics parser could not extract a first '
              f'token from {len(unparseable)} raw predictions; evaluated via '
              f'its existing empty-input branch. Sample IDs: {unparseable[:20]}. '
              'The unmodified raw outputs remain in prediction files.', flush=True)
    if blank:
        print(f'### Crossval warning: {len(blank)} empty/whitespace-only '
              f'generations. Sample IDs: {blank[:20]}. ',
              'Check EOS settings and the prediction file.', flush=True)
    metrics = process_metrics(annotations, metric_predictions)
    if not isinstance(metrics, dict):
        raise TypeError('process_metrics must return a metrics dictionary')
    metrics['empty_generation_count'] = len(blank)
    metrics['metrics_empty_parser_fallback_count'] = len(unparseable)
    metrics['metrics_empty_parser_fallback_ids'] = list(unparseable)
    return metrics


def evaluate_one_fold(model, base_dataset, val_indices, device, config, num_workers):
    val_subset = Subset(base_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=config.get("batch_size_test", config.get("batch_size_val", 1)),
        num_workers=num_workers,
        pin_memory=(getattr(model, "vision_device", torch.device("cpu")).type == "cuda"),
        shuffle=False,
        collate_fn=base_dataset.collate_fn,
        drop_last=False,
    )

    preds = []
    model.eval()
    with torch.inference_mode():
        for idx, vision_input, input_ids, input_atts in val_loader:
            # Leave batch tensors on CPU until generate() transfers each tensor
            # to the *actual* device of its consumer module.
            input_ids = input_ids[:, -128:]
            input_atts = input_atts[:, -128:]

            try:
                text_outputs = model.generate(
                    vision_input=vision_input,
                    input_ids=input_ids,
                    input_atts=input_atts,
                    use_nucleus_sampling=config.get("use_nucleus_sampling", False),
                    apply_lemmatizer=config["apply_lemmatizer"],
                    num_beams=config["num_beams"],
                    min_length=config["min_length"],
                    length_penalty=config.get("length_penalty", 1.0),
                    no_repeat_ngram_size=config.get("no_repeat_ngram_size", -1),
                    top_p=config.get("top_p", 0.9),
                    top_k=config.get("top_k", 3),
                    temperature=config.get("temperature", 1.0),
                    repetition_penalty=config.get("repetition_penalty", 1.0),
                    invalid_logits_policy=config.get("generation_invalid_logits_policy", "repair"),
                    max_new_tokens=config.get("max_new_tokens", 64),
                )
            except RuntimeError as exc:
                if 'Lynx numerical' not in str(exc):
                    raise
                raise RuntimeError(
                    f'Numerical generation failure evaluating sample IDs '
                    f'{list(idx)}. {exc}'
                ) from exc

            for i, output in zip(idx, text_outputs):
                preds.append({"index": i, "text_output": output.strip()})

            # Avoid per-batch empty_cache(): it adds synchronization overhead,
            # and does not reduce live tensor storage.

    annotations = [base_dataset.data[i] for i in val_indices]
    metrics = _process_metrics_with_empty_guard(annotations, preds)
    return preds, metrics


def aggregate_metrics(fold_metrics):
    keys = ["accuracy", "f1_fire", "f1_nofire", "f1_macro"]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=np.float64)
        agg[k] = {
            "mean": float(vals.mean()) if len(vals) else 0.0,
            "std": float(vals.std(ddof=0)) if len(vals) else 0.0,
        }
    return agg


def main(args, config):
    device = _resolve_requested_device(args.device, "device")
    llm_requested_device = _resolve_requested_device(args.llm_device, "llm_device")
    os.makedirs(args.output_dir, exist_ok=True)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = (device.type == "cuda")

    if args.dataset_files:
        config = copy.deepcopy(config)
        config["test_files"] = args.dataset_files

    print("### Creating dataset for cross-validation", flush=True)
    full_dataset = create_dataset("eval", config)
    n = len(full_dataset)
    assert args.k_folds >= 2, "k_folds must be >= 2"
    assert args.k_folds <= n, f"k_folds must be <= dataset size (n={n})"

    folds = make_kfold_indices(
        full_dataset.data,
        k=args.k_folds,
        seed=args.fold_seed,
        stratified=(not args.disable_stratified)
    )

    print(f"### Start {args.k_folds}-fold cross-validation with retraining", flush=True)
    start_cv = time.time()

    all_predictions = []
    all_fold_metrics = []
    fold_reports = []

    for fold_i, val_indices in enumerate(folds):
        fold_no = fold_i + 1
        val_set = set(val_indices)
        train_indices = [i for i in range(n) if i not in val_set]

        print(
            f"### Fold {fold_no}/{args.k_folds}: train={len(train_indices)} val={len(val_indices)}",
            flush=True,
        )

        fold_seed = seed + fold_i
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)

        # Clear any cyclic references from the previous fold before loading a
        # new 7B model. CUDA empty_cache alone does not release CPU objects.
        gc.collect()
        ram_memory_report(f"fold {fold_no} before model build")
        cuda_memory_report(f"fold {fold_no} before model build")

        model = build_model(
            config=config,
            device=device,
            llm_device=llm_requested_device,
            train_adapters=args.train_adapters
        )

        ram_memory_report(f"fold {fold_no} after model build")
        train_dataset = TrainFoldDataset(
            base_dataset=full_dataset,
            indices=train_indices,
            max_total_tokens=args.max_train_tokens,
            image_res=config.get('image_res', 224),
            image_mean=config.get('image_mean'),
            image_std=config.get('image_std'),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size_train,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            drop_last=False,
        )

        cuda_memory_report(f"fold {fold_no} after dataset and loader build")

        t0 = time.time()
        train_history = train_one_fold(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
        )
        train_time = time.time() - t0

        preds, metrics = evaluate_one_fold(
            model=model,
            base_dataset=full_dataset,
            val_indices=val_indices,
            device=device,
            config=config,
            num_workers=args.num_workers,
        )

        for p in preds:
            p["fold"] = fold_no
        all_predictions.extend(preds)
        all_fold_metrics.append(metrics)

        fold_report = {
            "fold": fold_no,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "train_time_sec": train_time,
            "train_history": train_history,
            "metrics": metrics,
        }
        fold_reports.append(fold_report)

        print(
            "### Fold {}/{} metrics: accuracy={:.4f}, f1_fire={:.4f}, f1_nofire={:.4f}, f1_macro={:.4f}".format(
                fold_no,
                args.k_folds,
                metrics["accuracy"],
                metrics["f1_fire"],
                metrics["f1_nofire"],
                metrics["f1_macro"],
            ),
            flush=True,
        )

        if args.save_fold_predictions:
            fold_pred_path = os.path.join(args.output_dir, f"fold_{fold_no}_predictions.jsonl")
            write_jsonl(preds, fold_pred_path)
            print(f"### Fold predictions saved to: {fold_pred_path}", flush=True)

        if args.save_fold_checkpoints:
            fold_ckpt_path = os.path.join(args.output_dir, f"fold_{fold_no}_model.pt")
            torch.save({"model": model.state_dict(), "config": config, "args": vars(args)}, fold_ckpt_path)
            print(f"### Fold model saved to: {fold_ckpt_path}", flush=True)

        # Release all fold-scoped references before allocating the next LLM.
        # Dataset workers and pinned buffers can otherwise survive to fold N+1.
        del train_loader, train_dataset, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ram_memory_report(f"fold {fold_no} after cleanup")

    cv_time = time.time() - start_cv
    aggregate = aggregate_metrics(all_fold_metrics)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_out = os.path.join(args.output_dir, f"cross_val_predictions_{timestamp}.jsonl")
    report_out = os.path.join(args.output_dir, f"cross_val_report_{timestamp}.json")

    write_jsonl(all_predictions, pred_out, 'w')

    report = {
        "k_folds": args.k_folds,
        "n_samples": n,
        "seed": args.seed,
        "fold_seed": args.fold_seed,
        "epochs": args.epochs,
        "batch_size_train": args.batch_size_train,
        "max_train_tokens": args.max_train_tokens,
        "cv_time": str(dt.timedelta(seconds=int(cv_time))),
        "aggregate_metrics": aggregate,
        "folds": fold_reports,
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("### Cross-validation finished", flush=True)
    print(f"### Predictions saved to: {pred_out}", flush=True)
    print(f"### Report saved to: {report_out}", flush=True)
    print(
        "### Aggregate metrics: accuracy={:.4f}±{:.4f}, f1_fire={:.4f}±{:.4f}, f1_nofire={:.4f}±{:.4f}, f1_macro={:.4f}±{:.4f}".format(
            aggregate["accuracy"]["mean"],
            aggregate["accuracy"]["std"],
            aggregate["f1_fire"]["mean"],
            aggregate["f1_fire"]["std"],
            aggregate["f1_nofire"]["mean"],
            aggregate["f1_nofire"]["std"],
            aggregate["f1_macro"]["mean"],
            aggregate["f1_macro"]["std"],
        ),
        flush=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Vision device: cpu, gpu, cuda or cuda:N")
    parser.add_argument("--llm_device", type=str, default="cpu",
                        help="LLM requested device: cpu, gpu, cuda or cuda:N")
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--data_files", type=str, nargs="*", default=None)
    parser.add_argument("--dataset_files", nargs="+", default=None)
    parser.add_argument("--k_folds", type=int, default=5, required=True)
    parser.add_argument("--fold_seed", type=int, default=123)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5) #2e-5
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", default=2, type=int)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_tokens", type=int, default=256) # 128
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--output_predictions", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./cv_outputs")
    # parser.add_argument("--report_path", type=str, default="cross_val_report.json")
    parser.add_argument("--save_fold_predictions", action="store_true")
    parser.add_argument("--save_fold_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./cv_checkpoints")

    parser.add_argument("--train_adapters", action="store_true")
    
    parser.add_argument("--disable_stratified", action="store_true")
    #parser.add_argument("--num_workers", default=4, type=int)
    
    args = parser.parse_args()

    if args.output_predictions is None:
        current_datetime = datetime.now()
        args.output_predictions = (
            f"./cv_predictions_{current_datetime.day}_{current_datetime.month}-"
            f"{current_datetime.hour}_{current_datetime.minute}.jsonl"
        )

    yaml = YAML(typ="rt")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    main(args, config) #import argparse