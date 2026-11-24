# Copyright (2023) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import math
import json
import os
import re
import sys
import warnings
from pathlib import Path
from torch.utils.checkpoint import checkpoint as _torch_activation_checkpoint

import torch
import torch.nn as nn
import torch.distributed as dist

from einops import rearrange

from timm.models.layers import trunc_normal_

# Prefer the tokenizer helper bundled with this model package. Some project
# layouts instead put additional.py at the repository root. Importing from the
# package first avoids requiring users to modify sys.path in cross_val.py.
try:
    from .additional import build_tokenizer
except ModuleNotFoundError as exc:
    if exc.name != f"{__package__}.additional":
        # The helper exists, but one of its OWN dependencies is missing.
        raise
    try:
        from additional import build_tokenizer
    except ModuleNotFoundError as exc:
        if exc.name != 'additional':
            raise

        def build_tokenizer(llm_name, use_left_pad=False):
            """Compatible fallback when the project's additional.py is absent.

            Uses the existing *local* tokenizer files. Does not download model
            files or add vocabulary entries unnecessarily; this preserves the
            pretrained LLaMA embedding size whenever possible.

            If a project relies on custom tokenizer/special-token behavior in
            additional.py, restore that helper rather than using this fallback.
            """
            try:
                from transformers import AutoTokenizer
            except ImportError as missing:
                raise ImportError(
                    'No additional.py tokenizer helper was found, and the '
                    'transformers package needed by the fallback is missing. '
                    'Install transformers or restore additional.py in models/ '
                    'or at the project root.'
                ) from missing

            project_root = Path(__file__).resolve().parent.parent
            candidate_dirs = (
                project_root / 'data' / llm_name,
                Path('data') / llm_name,
            )
            model_dir = next((p for p in candidate_dirs if p.is_dir()), None)
            if model_dir is None:
                raise FileNotFoundError(
                    'Tokenizer files not found in data/' + str(llm_name)
                    + '. Restore additional.py or put the Vicuna tokenizer '
                    'files under the project data/ directory.'
                )

            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), use_fast=False, local_files_only=True
            )
            tokenizer.padding_side = 'left' if use_left_pad else 'right'
            added = 0
            if tokenizer.pad_token_id is None:
                # Conventional LLaMA fallback: re-use an existing token to
                # avoid resizing the pretrained embedding table.
                if tokenizer.unk_token_id is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                elif tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    added = tokenizer.add_special_tokens(
                        {'pad_token': '[PAD]'}
                    )
            print('### Using local Hugging Face tokenizer fallback '
                  f'({model_dir}); newly added tokens: {added}', flush=True)
            return tokenizer, added

from .vits.eva_vit import interpolate_pos_embed


# Verify this exact source is active with:
#   import models.lynx as lynx; print(lynx.LYNX_GENERATION_PATCH_VERSION)
LYNX_GENERATION_PATCH_VERSION = 'direct-legacy-kv-stable-decoder-v8-2026-11-23'
LYNX_TRAIN_NUMERICS_PATCH_VERSION = "sequential-zero-copy-activation-checkpoint-v18-2026-11-24"
LYNX_PREATTENTION_PATCH_VERSION = "rope-buffer-rmsnorm-preprojection-v17-2026-11-23"
LYNX_ROPE_MASK_PATCH_VERSION = "legacy-rope-binary-mask-v12-2026-11-23"
LYNX_CUDA_MEMORY_PATCH_VERSION = 'cuda-llm-submodule-offload-guard-v6-2026-11-23'
LYNX_CROSSVAL_INTEGRATION_VERSION = 'crossval-stable-sampling-v8-2026-11-23'
LYNX_NUMERICS_PATCH_VERSION = 'legacy-kv-stable-logits-filter-v8-2026-11-23'





# Legacy Lynx LLaMA calculates its rotary table for the *actual* token count.
# Its original mask->positions code assumed masks contain only 0 and 1.
# A non-binary bridge mask makes cumulative positions grow beyond the rotary
# cache (for example a 46-token prefill produces position 1404), causing:
# RuntimeError: index 1404 is out of bounds for dimension 2 with size 46.
_LYNX_NONBINARY_MASK_WARNINGS = set()
_LYNX_RESAMPLER_MASK_LOGGED = False


def _lynx_normalize_attention_mask(mask, *, name='attention_mask'):
    """Return a binary int64 attention mask, validating input semantics.

    The attention mask is an *indicator*, not a token count or a position.
    Positive values mean 'attend'; zero means padding.  Report masks with
    values >1 once per source so incorrect upstream bridge masks remain visible.
    Negative, non-finite, or malformed masks are rejected rather than guessed.
    """
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.numel() == 0:
        raise ValueError(f'Lynx {name} must be a nonempty [batch, tokens] tensor')
    if mask.is_complex():
        raise ValueError(f'Lynx {name} must be a real-valued binary attention mask')
    if not bool(torch.isfinite(mask).all()):
        raise ValueError(f'Lynx {name} contains NaN or Inf; cannot construct RoPE positions')
    if bool((mask < 0).any()):
        raise ValueError(f'Lynx {name} contains negative values; expected 0 or positive indicators')
    nonbinary = bool(((mask != 0) & (mask != 1)).any())
    if nonbinary and name not in _LYNX_NONBINARY_MASK_WARNINGS:
        warnings.warn(
            f'Lynx {name} contains non-binary values (max={mask.max().item():g}); '
            'normalizing positive entries to 1 before RoPE position calculation. '
            'Check the dataset/bridge attention-mask output.',
            RuntimeWarning, stacklevel=2,
        )
        _LYNX_NONBINARY_MASK_WARNINGS.add(name)
    if not nonbinary and mask.dtype == torch.long:
        return mask
    return (mask > 0).to(dtype=torch.long)


def _lynx_resampler_attention_mask(latent_features, reported_mask=None, *, name='resampler'):
    """Build an attention mask from the actual learned Perceiver latent outputs.

    The Lynx PerceiverResampler returns a *fixed number of active latent
    queries*. They are not padded, so every output query is attendable. The
    second bridge return value has sometimes contained arbitrary non-binary
    values (e.g. 29973), which are not valid attention indicators. Never
    interpret those values as token counts or RoPE offsets; use the semantics
    of the fixed-latent bridge instead. Validate its output contract first.

    This helper is specifically for the resampler bridge. Other future bridge
    types with variable/padded query counts must use their own real masks.
    """
    global _LYNX_RESAMPLER_MASK_LOGGED
    if not isinstance(latent_features, torch.Tensor) or latent_features.ndim != 3:
        raise ValueError(f'Lynx {name} latent features must be [batch, query, width]')
    if latent_features.size(0) < 1 or latent_features.size(1) < 1:
        raise ValueError(f'Lynx {name} returned zero latent queries')
    if reported_mask is not None:
        if not isinstance(reported_mask, torch.Tensor) or reported_mask.ndim != 2:
            raise ValueError(f'Lynx {name} returned a malformed attention mask')
        if tuple(reported_mask.shape) != tuple(latent_features.shape[:2]):
            raise ValueError(
                f'Lynx {name} mask shape {tuple(reported_mask.shape)} does not match '
                f'latent output {tuple(latent_features.shape[:2])}'
            )
    if not _LYNX_RESAMPLER_MASK_LOGGED:
        print('### Lynx resampler: all learned latent queries are active; '
              'building a binary mask from latent output shape (ignoring '
              'bridge-provided mask values).', flush=True)
        _LYNX_RESAMPLER_MASK_LOGGED = True
    return torch.ones(latent_features.shape[:2], device=latent_features.device,
                      dtype=torch.long)


def _lynx_positions_from_mask(mask, *, current_length, past_length=0,
                              name='attention_mask'):
    """Compute RoPE positions and assert every index fits the KV length.

    In a legacy LLaMA, the rotary cache is built with
    kv_seq_len = past_length + current_length.  Position IDs MUST be within
    [0, kv_seq_len).  This holds for left-padding and beam-expanded batches.
    """
    mask = _lynx_normalize_attention_mask(mask, name=name)
    current_length = int(current_length)
    past_length = int(past_length)
    if current_length <= 0 or past_length < 0:
        raise ValueError('Lynx current_length must be positive, past_length nonnegative')
    kv_length = past_length + current_length
    if mask.size(1) != kv_length:
        raise ValueError(
            f'Lynx {name} length {mask.size(1)} differs from expected '
            f'cache+tokens length {kv_length} (past={past_length}, current={current_length})'
        )
    positions = mask.cumsum(dim=-1) - 1
    # An entirely masked position must still have a valid RoPE index.
    positions.masked_fill_(mask == 0, 0)
    positions = positions[:, -current_length:].contiguous()
    if bool((positions < 0).any()) or bool((positions >= kv_length).any()):
        raise ValueError(
            f'Lynx RoPE positions out of bounds: allowed [0, {kv_length - 1}], '
            f'actual [{positions.min().item()}, {positions.max().item()}]'
        )
    return positions


def _valid_vocab_id(value, vocab_size):
    """Return True for integer special-token IDs that exist in this vocabulary."""
    return (isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value < int(vocab_size))


def _ensure_valid_tokenizer_pad(tokenizer):
    """Repair a missing/-1 pad ID by reusing a real special token.

    Do not add a new token solely to provide padding: the custom Vicuna model
    and its fine-tuned checkpoint normally use a fixed 32000-token vocabulary.
    """
    tokenizer_length = len(tokenizer)
    if tokenizer_length <= 0:
        raise ValueError('Tokenizer must contain at least one token')
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if _valid_vocab_id(pad_id, tokenizer_length):
        return pad_id

    for kind in ('eos', 'unk', 'bos'):
        candidate_id = getattr(tokenizer, kind + '_token_id', None)
        if not _valid_vocab_id(candidate_id, tokenizer_length):
            continue
        token_string = getattr(tokenizer, kind + '_token', None)
        if token_string is None and hasattr(tokenizer, 'convert_ids_to_tokens'):
            token_string = tokenizer.convert_ids_to_tokens(candidate_id)
        if not token_string:
            continue
        tokenizer.pad_token = token_string
        updated_id = getattr(tokenizer, 'pad_token_id', None)
        if _valid_vocab_id(updated_id, tokenizer_length):
            print(f'### Lynx tokenizer: invalid padding ID {pad_id!r}; '
                  f'reusing {kind}_token (ID {updated_id}) without resizing vocab',
                  flush=True)
            return updated_id

    raise ValueError(
        f'Could not obtain a valid tokenizer pad_token_id (received {pad_id!r}, '
        f'tokenizer length={tokenizer_length}). Check additional.py, tokenizer '
        'special tokens and data/vicuna-7b tokenizer files.'
    )


def _adapter_layer_indices(names):
    """Extract layer indices of output-adapter weights from model/ckpt keys."""
    indices = set()
    for name in names:
        match = re.search(r'(?:^|\.)layers\.(\d+)\.output_adapter\.', name)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _adapter_stride_hint(checkpoint_layers):
    """Infer adapter frequency only when every observed layer gap agrees."""
    if len(checkpoint_layers) < 2:
        return None
    steps = {b - a for a, b in zip(checkpoint_layers, checkpoint_layers[1:])}
    if len(steps) == 1:
        return steps.pop()
    return None




def _lynx_legacy_sdpa_keep_mask(mask, *, batch, query_length, key_length, device):
    """Translate legacy LLaMA's additive causal/padding mask to SDPA keep-bits.

    Original Lynx constructs a 4-D additive attention mask with 0 at allowed
    keys and a very negative value (e.g. -1e4, finfo.min, -inf) at blocked
    keys. Passing this directly to SDPA can be fragile with fully masked rows,
    particularly when multiple very negative masks have been summed. PyTorch
    SDPA's BOOLEAN convention is the opposite of MultiheadAttention's mask:
    True means the key is allowed. Preserve that interpretation explicitly.
    """
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError('Lynx attention mask must be a tensor')
    shape = (batch, 1, query_length, key_length)
    if tuple(mask.shape) != shape:
        raise ValueError(f'Lynx attention mask: got {tuple(mask.shape)}, expected {shape}')
    if mask.dtype == torch.bool:
        return mask.to(device=device)
    if not torch.is_floating_point(mask):
        raise TypeError('Lynx 4-D attention mask must be floating-point additive or bool')
    if bool(torch.isnan(mask).any()) or bool(torch.isposinf(mask).any()):
        raise ValueError('Legacy additive attention mask has NaN/+Inf')
    if bool((mask > 1e-4).any()):
        raise ValueError('Unexpected positive bias in legacy additive attention mask')
    # This 2023 LLaMA implementation has no learned attention bias. Reject
    # intermediate negative biases rather than silently changing semantics.
    ambiguous = (mask < -1e-4) & (mask > -1e3)
    if bool(ambiguous.any()):
        raise ValueError('Unsupported intermediate additive attention bias; '
                         'expected 0 or a large negative masking sentinel')
    return (mask >= -1e-4).to(device=device)


def _lynx_masked_reference_attention(q, k, v, keep):
    """Rare numerical fallback, using FP64 scores and exact allowed keys.

    Runs only if SDPA yields a non-finite result despite finite Q/K/V. It does
    NOT conceal invalid source activations or weights. Fully masked queries
    have zero attention output, as in PyTorch's boolean SDPA semantics.
    """
    import math
    query = q.double()
    key = k.double()
    value = v.double()
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if keep is not None:
        scores = scores.masked_fill(~keep, float('-inf'))
        all_blocked = ~keep.any(dim=-1, keepdim=True)
        # Prevent an all -Inf row from generating NaNs in softmax.
        scores = scores.masked_fill(all_blocked, 0.0)
    else:
        all_blocked = None
    probs = torch.softmax(scores, dim=-1)
    if all_blocked is not None:
        probs = probs.masked_fill(all_blocked, 0.0)
    return (probs @ value).to(q.dtype)


def _lynx_refresh_rotary_state(rotary, *, head_dim, repair=True, theta=10000.0,
                               cache_check=True, label='rotary'):
    """Validate/reconstruct ONLY procedural RoPE buffers (never weight tensors).

    Lynx's original 2023 LlamaRotaryEmbedding registers `inv_freq` as a
    persistent buffer; the base-weight finite preflight previously inspected
    parameters but not buffers. NaN frequencies can make all projected queries
    non-finite during RoPE despite finite checkpoint PARAMETERS.

    The formula below matches that original class (base=10000 by default).
    Arbitrary learned parameters, adapters and checkpoint weights are untouched.
    """
    inv_freq = getattr(rotary, 'inv_freq', None)
    if not isinstance(inv_freq, torch.Tensor):
        return (False, False)  # e.g. a different or synthetic rotary module
    if not inv_freq.is_floating_point() or inv_freq.ndim != 1 or inv_freq.numel() != head_dim // 2:
        raise RuntimeError(f'{label}: invalid RoPE inv_freq shape/dtype {tuple(inv_freq.shape)}')
    if not (float(theta) > 0 and math.isfinite(float(theta))):
        raise ValueError(f'{label}: invalid RoPE theta={theta}')
    with torch.no_grad():
        good_freq = bool(torch.isfinite(inv_freq).all() and (inv_freq > 0).all())
        repaired_freq = False
        if not good_freq:
            if not repair:
                raise RuntimeError(f'{label}: invalid RoPE inv_freq buffer; '
                                   'set legacy_llama_repair_rope_buffers: true or repair the base checkpoint')
            # The original LLaMA rotary class uses this deterministic formula.
            idx = torch.arange(0, head_dim, 2, dtype=torch.float64, device=inv_freq.device)
            expected = (float(theta) ** (-idx / head_dim)).to(dtype=inv_freq.dtype)
            if not bool(torch.isfinite(expected).all()):
                raise RuntimeError(f'{label}: computed RoPE frequencies are invalid')
            inv_freq.copy_(expected)
            repaired_freq = True
        n = int(getattr(rotary, 'max_seq_len_cached', 0))
        if n < 1:
            raise RuntimeError(f'{label}: invalid cached RoPE length {n}')
        cos_old, sin_old = getattr(rotary, 'cos_cached', None), getattr(rotary, 'sin_cached', None)
        healthy_cache = (isinstance(cos_old, torch.Tensor) and isinstance(sin_old, torch.Tensor)
                         and cos_old.shape == sin_old.shape == (1, 1, n, head_dim))
        if healthy_cache and cache_check:
            healthy_cache = bool(torch.isfinite(cos_old).all() and torch.isfinite(sin_old).all())
            if healthy_cache:
                # Guard against stale caches after loading a checkpoint's inv_freq.
                # Only sample a few positions, avoiding a second full cache copy.
                checks = sorted({0, min(1, n - 1), n // 2, n - 1})
                check_t = torch.tensor(checks, device=inv_freq.device, dtype=torch.float32)
                sampled = torch.outer(check_t, inv_freq.float())
                sampled = torch.cat((sampled, sampled), dim=-1)
                reference_cos = sampled.cos().to(cos_old.dtype)
                reference_sin = sampled.sin().to(sin_old.dtype)
                healthy_cache = (torch.allclose(cos_old[0, 0, checks], reference_cos,
                                                atol=5e-4, rtol=5e-4)
                                 and torch.allclose(sin_old[0, 0, checks], reference_sin,
                                                    atol=5e-4, rtol=5e-4))
        if repaired_freq or not healthy_cache:
            if not repair:
                raise RuntimeError(f'{label}: RoPE cosine/sine buffers are invalid or stale')
            times = torch.arange(n, device=inv_freq.device, dtype=inv_freq.dtype)
            freq_grid = torch.outer(times, inv_freq)
            embeddings = torch.cat((freq_grid, freq_grid), dim=-1)
            cosine, sine = embeddings.cos()[None, None], embeddings.sin()[None, None]
            if not bool(torch.isfinite(cosine).all() and torch.isfinite(sine).all()):
                raise RuntimeError(f'{label}: cannot recover finite RoPE tables from inv_freq; '
                                   'inspect the source checkpoint or rotary configuration')
            rotary.register_buffer('cos_cached', cosine, persistent=False)
            rotary.register_buffer('sin_cached', sine, persistent=False)
            return (repaired_freq, True)
    return (False, False)


def _lynx_validate_llama_rotary_buffers(llm, *, repair=True):
    """Preflight every rotary buffer *after* checkpoint loading, before CUDA."""
    layers = getattr(getattr(llm, 'model', None), 'layers', ())
    repaired, rebuilt, scanned = 0, 0, 0
    for i, layer in enumerate(layers):
        attn = getattr(layer, 'self_attn', None)
        rotary = getattr(attn, 'rotary_emb', None)
        if rotary is None or not hasattr(rotary, 'inv_freq'):
            continue
        scanned += 1
        theta = float(getattr(rotary, 'base', 10000.0))
        a, b = _lynx_refresh_rotary_state(
            rotary, head_dim=attn.head_dim, repair=repair, theta=theta,
            label=f'LLM.model.layers.{i}.self_attn.rotary_emb')
        repaired += int(a)
        rebuilt += int(b)
    print(f'### Lynx rotary preflight: {scanned} layers, '
          f'{repaired} invalid frequency buffers reconstructed, '
          f'{rebuilt} stale/invalid caches rebuilt', flush=True)
    return {'layers': scanned, 'repaired_frequencies': repaired, 'rebuilt_caches': rebuilt}


def _lynx_enable_stable_rmsnorm(llm, *, enabled=True):
    """Numerically stable RMSNorm for large but finite multimodal embeddings.

    Scaling each token by max(1, max(abs(x))) before squaring prevents FP32
    overflow in variance. Mathematically equivalent to the original RMSNorm
    for ordinary values, with gradients intact. Never changes trained weights.
    """
    if not enabled:
        return 0
    import types

    def stable_norm(self, hidden_states):
        if not bool(torch.isfinite(hidden_states).all()):
            raise FloatingPointError(f'{self._lynx_norm_name}: non-finite RMSNorm input')
        eps = float(self.variance_epsilon)
        if not (math.isfinite(eps) and eps > 0):
            raise FloatingPointError(f'{self._lynx_norm_name}: invalid RMSNorm epsilon={eps}')
        values = hidden_states.float()
        magnitude = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        scaled = values / magnitude
        normalized = scaled * torch.rsqrt(scaled.square().mean(dim=-1, keepdim=True)
                                          + (eps / magnitude.square()))
        # Match original Lynx behavior of converting to weight dtype if half.
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            normalized = normalized.to(dtype=self.weight.dtype)
        output = normalized * self.weight
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError(f'{self._lynx_norm_name}: RMSNorm output is '
                                     'non-finite despite finite input; check norm weights')
        return output

    root = getattr(llm, 'model', None)
    if root is None:
        return 0
    names = [(f'layers.{i}.{kind}', getattr(layer, kind, None))
             for i, layer in enumerate(getattr(root, 'layers', ()))
             for kind in ('input_layernorm', 'post_attention_layernorm')]
    names.append(('final_norm', getattr(root, 'norm', None)))
    count = 0
    for name, norm in names:
        if norm is None or getattr(norm, '_lynx_stable_rmsnorm', False):
            continue
        if not hasattr(norm, 'variance_epsilon') or not hasattr(norm, 'weight'):
            continue
        norm._lynx_norm_name = name
        norm.forward = types.MethodType(stable_norm, norm)
        norm._lynx_stable_rmsnorm = True
        count += 1
    print(f'### Lynx stable RMSNorm: enabled on {count} modules '
          '(scale-before-square, checkpoint weights unchanged)', flush=True)
    return count


def _lynx_enable_stable_legacy_attention(llm, *, backend='sdpa', check_qkv=True, repair_rope=True):
    """Use boolean-mask SDPA with finite-QKV checks for custom Lynx LLaMA.

    The checkpoint parameters, adapter modules, RoPE and tuple KV cache are
    unchanged. Patch only this model instance; no global torch monkeypatch.
    A fully masked query returns zero attention (not a NaN or uniform average
    of masked values). A kernel-generated NaN with finite QKV triggers a
    differentiable float64 reference calculation rather than fake logits.
    """
    if backend in ('legacy', 'original', 'off'):
        print('### Lynx attention backend: original legacy attention (may reproduce NaNs)', flush=True)
        return 0
    if backend != 'sdpa':
        raise ValueError('legacy_llama_attention_backend must be sdpa or legacy')

    import types
    import torch.nn.functional as F
    from models.llms.llama import modeling_llama as legacy_llama
    rotary = getattr(legacy_llama, 'apply_rotary_pos_emb', None)
    if rotary is None or not callable(rotary):
        raise RuntimeError('Cannot patch Lynx: legacy RoPE function missing')
    layers = getattr(getattr(llm, 'model', None), 'layers', None)
    if layers is None:
        raise RuntimeError('Cannot patch Lynx: LLM.model.layers missing')

    def stable_attention(self, hidden_states, attention_mask=None, position_ids=None,
                         past_key_value=None, output_attentions=False, use_cache=False):
        if hidden_states.ndim != 3:
            raise ValueError('Lynx attention expects [batch, query_length, width]')
        bsz, q_len, _ = hidden_states.shape
        nh, hd = self.num_heads, self.head_dim
        layer_id = getattr(self, '_lynx_attention_layer_id', '?')
        if not bool(torch.isfinite(hidden_states).all()):
            raise FloatingPointError(f'LLM layer {layer_id}: query/key/value input '
                                     'hidden states are already non-finite BEFORE projections')
        projections = []
        for name, module in (('query', self.q_proj), ('key', self.k_proj), ('value', self.v_proj)):
            projected = module(hidden_states)
            if check_qkv and not bool(torch.isfinite(projected).all()):
                input_scale = float(hidden_states.detach().abs().amax().item())
                weight_scale = float(module.weight.detach().abs().amax().item())
                raise FloatingPointError(
                    f'LLM layer {layer_id}: {name} projection produced NaN/Inf '
                    f'BEFORE rotary embeddings; max_abs_input={input_scale:.5g}, '
                    f'max_abs_weight={weight_scale:.5g}. Check input_layernorm and '
                    'pretrained projection weights or unusual activation magnitudes.')
            projections.append(projected.view(bsz, q_len, nh, hd).transpose(1, 2))
        q, k, v = projections
        past_len = 0
        if past_key_value is not None:
            if (not isinstance(past_key_value, (tuple, list))
                    or len(past_key_value) != 2):
                raise TypeError('Lynx attention expects a legacy (key, value) cache')
            past_len = past_key_value[0].size(-2)
            if (past_key_value[0].shape[:2] != (bsz, nh)
                    or past_key_value[1].shape[:2] != (bsz, nh)):
                raise ValueError('Lynx attention KV cache batch/head mismatch')
        kv_len = past_len + q_len
        if position_ids is None:
            position_ids = torch.arange(past_len, kv_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        if tuple(position_ids.shape) != (bsz, q_len):
            raise ValueError('Lynx attention position_ids do not match current tokens')
        cos, sin = self.rotary_emb(v, seq_len=kv_len)
        if not bool(torch.isfinite(cos).all() and torch.isfinite(sin).all()):
            # The original LLaMA stores inv_freq as a persistent buffer. A bad
            # base checkpoint buffer is not caught by a parameter-only scan.
            if getattr(self, '_lynx_repair_rope_buffers', True):
                changed = _lynx_refresh_rotary_state(
                    self.rotary_emb, head_dim=hd,
                    repair=True, theta=float(getattr(self.rotary_emb, 'base', 10000.0)),
                    label=f'LLM layer {layer_id}.rotary_emb')
                if any(changed):
                    print(f'### Lynx: regenerated invalid RoPE buffer in layer {layer_id}', flush=True)
                cos, sin = self.rotary_emb(v, seq_len=kv_len)
            if not bool(torch.isfinite(cos).all() and torch.isfinite(sin).all()):
                raise FloatingPointError(f'LLM layer {layer_id}: non-finite RoPE '
                                         'cos/sin buffers BEFORE attention')
        q, k = rotary(q, k, cos, sin, position_ids)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)
        present = (k, v) if use_cache else None

        if check_qkv:
            # Errors originating before attention should not be hidden by a
            # mask change or nan_to_num. Identify the first corrupted tensor.
            for tensor, name in ((q, 'query'), (k, 'key'), (v, 'value')):
                if not bool(torch.isfinite(tensor).all()):
                    invalid_positions = torch.nonzero(
                        (~torch.isfinite(tensor)).any(dim=(1, 3)), as_tuple=False
                    )[:12].tolist()
                    raise FloatingPointError(
                        f'LLM layer {layer_id}: {name} became NaN/Inf '
                        f'AFTER RoPE/before SDPA (raw projections were finite); '
                        f'batch_token_positions={invalid_positions}; '
                        f'shape={tuple(tensor.shape)}. Inspect rotary frequencies '
                        'and the magnitude of projected activations.'
                    )

        keep = _lynx_legacy_sdpa_keep_mask(
            attention_mask, batch=bsz, query_length=q_len,
            key_length=kv_len, device=q.device
        )
        if keep is None:
            # Explicitly maintain causal semantics even for cached multi-token
            # forward passes, where is_causal=True can be misaligned.
            row = torch.arange(q_len, device=q.device)[:, None] + past_len
            col = torch.arange(kv_len, device=q.device)[None, :]
            keep = (col <= row)[None, None, :, :]
        # PyTorch SDPA: boolean True means ATTEND. A fully masked query emits
        # zero attention instead of softmax(-Inf, -Inf, ...) -> NaNs.
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep, dropout_p=0.0, is_causal=False
        )
        if not bool(torch.isfinite(output).all()):
            import warnings
            if not getattr(self, '_lynx_sdpa_reference_warned', False):
                warnings.warn('Lynx SDPA produced a non-finite value with finite Q/K/V; '
                              'retrying the attention scores in FP64.', RuntimeWarning)
                self._lynx_sdpa_reference_warned = True
            output = _lynx_masked_reference_attention(q, k, v, keep)
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError(
                    'Lynx attention still non-finite after FP64 masked reference; '
                    'check Q/K/V magnitudes and numerical range.'
                )
        y = output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        output = self.o_proj(y)
        weights = None
        if output_attentions:
            scores = (q.float() @ k.float().transpose(-2, -1)) * (hd ** -0.5)
            scores = scores.masked_fill(~keep, float('-inf'))
            all_masked = ~keep.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_masked, 0.0)
            weights = torch.softmax(scores, dim=-1).masked_fill(all_masked, 0.0).to(q.dtype)
        return output, weights, present

    count = 0
    for i, layer in enumerate(layers):
        attention = getattr(layer, 'self_attn', None)
        if attention is None:
            raise RuntimeError(f'Legacy attention missing at decoder layer {i}')
        if getattr(attention, '_lynx_stable_sdpa', False):
            continue
        required = ('num_heads', 'head_dim', 'hidden_size', 'q_proj', 'k_proj',
                    'v_proj', 'o_proj', 'rotary_emb')
        missing = [name for name in required if not hasattr(attention, name)]
        if missing:
            raise RuntimeError(f'Unsupported legacy attention layer {i}: missing {missing}')
        attention.forward = types.MethodType(stable_attention, attention)
        attention._lynx_stable_sdpa = True
        attention._lynx_attention_v17 = True
        attention._lynx_attention_layer_id = i
        attention._lynx_repair_rope_buffers = bool(repair_rope)
        count += 1
    print(f'### Lynx attention backend: rope-diagnostic-sdpa-v17 applied to {count} '
          'legacy LLaMA layers (no checkpoint or adapter changes)', flush=True)
    return count


def _with_generation_mixin(model_class):
    """Compatibility shim for the project's legacy LLaMA and current Transformers.

    This model uses tuple-of-tuples KV caches, not HF DynamicCache. Its original
    prepare_inputs_for_generation() also trusts a cached `position_ids` kwarg.
    Recent HF versions sometimes preserve the *full prompt* position_ids after
    the first decode step. In the original LlamaModel.forward(), that full
    tensor is reshaped with `.view(-1, seq_length)`, silently turning the
    previous sequence length into a batch dimension during one-token decoding.
    The resulting key/value batch mismatch causes torch.cat(..., dim=2) to fail.

    Override the preparation method to ALWAYS recompute batch-aligned position
    IDs from the current attention mask and slice to the current step length.
    Keep tuple-based caching and both CPU/GPU loading paths intact.
    """
    import functools
    from transformers.generation import GenerationMixin

    if getattr(model_class, '_lynx_legacy_cache_wrapper', False):
        return model_class

    bases = ((model_class,) if issubclass(model_class, GenerationMixin)
             else (model_class, GenerationMixin))

    @functools.wraps(model_class.forward)
    def guarded_legacy_forward(self, *args, **kwargs):
        """Protect legacy RoPE from HF-supplied full-prompt position IDs.

        In the original LlamaModel.forward(), position_ids.view(-1, seq_length)
        can silently turn 47 positions into 47 *batch elements* when the new
        token has seq_length=1. The resulting RoPE broadcast expands K from
        batch 1 to 47 and breaks cached attention. This guard validates the
        tensor shapes immediately before the real legacy model's forward.

        Normally prepare_inputs_for_generation has already fixed the IDs; this
        is a second line of defense for library-specific generation paths.
        """
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        inputs_embeds = kwargs.get('inputs_embeds')
        active = input_ids if input_ids is not None else inputs_embeds
        if active is None:
            return model_class.forward(self, *args, **kwargs)
        if active.ndim not in (2, 3):
            raise ValueError(f'Invalid legacy LLaMA input shape: {tuple(active.shape)}')

        batch_size, seq_length = active.shape[:2]
        cache = kwargs.get('past_key_values')
        if cache is not None:
            if not isinstance(cache, (tuple, list)):
                raise TypeError('Legacy Lynx LLaMA requires tuple/list KV caches; '
                                f'got {type(cache).__name__}')
            if cache and cache[0][0].shape[0] != batch_size:
                raise ValueError('KV-cache batch size does not match current input: '
                                 f'cache={cache[0][0].shape[0]}, input={batch_size}. '
                                 'Check beam expansion/cache reordering.')

        past_length = cache[0][0].size(2) if cache else 0
        mask = kwargs.get('attention_mask')
        if mask is not None:
            mask = _lynx_normalize_attention_mask(mask, name='llm_forward_attention_mask')
            if mask.size(0) != batch_size:
                raise ValueError('Lynx LLM attention mask batch size differs from input batch')
            kwargs['attention_mask'] = mask
            expected_positions = _lynx_positions_from_mask(
                mask, current_length=seq_length, past_length=past_length,
                name='llm_forward_attention_mask',
            )
        else:
            expected_positions = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long,
                device=active.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # Recompute for every call, even if the supplied position_ids has the
        # correct *shape*: values can still be > available rotary cache length.
        original_positions = kwargs.get('position_ids')
        if (original_positions is not None and
                (original_positions.shape != expected_positions.shape or
                 not torch.equal(original_positions, expected_positions)) and
                not getattr(self, '_lynx_rope_positions_repaired_logged', False)):
            print('### Lynx: corrected invalid/stale RoPE position_ids before LLM forward',
                  flush=True)
            self._lynx_rope_positions_repaired_logged = True
        kwargs['position_ids'] = expected_positions

        return model_class.forward(self, *args, **kwargs)

    def supports_modern_cache(cls):
        # Instruct GenerationMixin not to construct a DynamicCache instance.
        return False

    def prepare_legacy_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs
    ):
        """Produce correctly sized legacy inputs for prefill and next-token steps.

        Do not pass HF-only `cache_position` to this legacy forward(). Ignore
        potentially stale kwargs['position_ids'] instead of slicing them:
        after prefill, those values may omit the newly generated token's index.
        """
        if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
            raise TypeError(
                'Lynx legacy LLaMA requires tuple-based past_key_values; '
                f'got {type(past_key_values).__name__}. Disable modern HF caches.'
            )

        has_past = past_key_values is not None and len(past_key_values) > 0
        if has_past:
            if input_ids is None or input_ids.ndim != 2 or input_ids.size(1) == 0:
                raise ValueError('Cached generation requires at least one input token')
            # All earlier tokens are already represented by the KV cache.
            model_inputs = {'input_ids': input_ids[:, -1:]}
            current_length = 1
        elif inputs_embeds is not None:
            if inputs_embeds.ndim != 3:
                raise ValueError('inputs_embeds must have shape [batch, tokens, hidden]')
            model_inputs = {'inputs_embeds': inputs_embeds}
            current_length = inputs_embeds.size(1)
        elif input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError('input_ids must have shape [batch, tokens]')
            model_inputs = {'input_ids': input_ids}
            current_length = input_ids.size(1)
        else:
            raise ValueError('Generation requires input_ids or inputs_embeds')

        active_tensor = next(iter(model_inputs.values()))
        batch_size = active_tensor.size(0)
        if attention_mask is not None:
            if (attention_mask.ndim != 2 or attention_mask.size(0) != batch_size
                    or attention_mask.size(1) < current_length):
                raise ValueError(
                    'Invalid generation attention_mask shape: '
                    f'expected [batch={batch_size}, length>={current_length}], '
                    f'got {tuple(attention_mask.shape)}'
                )
            # Fresh positions are essential. HF may pass stale position_ids
            # from the full prompt on a one-token decoding step; do not reuse.
            attention_mask = _lynx_normalize_attention_mask(
                attention_mask, name='hf_prepare_attention_mask')
            previous = past_key_values[0][0].size(2) if has_past else 0
            position_ids = _lynx_positions_from_mask(
                attention_mask, current_length=current_length,
                past_length=previous, name='hf_prepare_attention_mask')
        else:
            # Without a mask, advance from the length represented by the cache.
            offset = past_key_values[0][0].size(-2) if has_past else 0
            position_ids = torch.arange(
                offset, offset + current_length, dtype=torch.long,
                device=active_tensor.device
            ).unsqueeze(0).expand(batch_size, -1)

        model_inputs.update({
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache', True),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def _is_cuda_destination(args, kwargs):
        """Check the device-only forms of nn.Module.to without changing dtype calls.

        Handles .to('cuda'), .to(device=torch.device('cuda')), and
        .to(cuda_tensor). If only a dtype is provided, the normal .to runs.
        """
        target = kwargs.get('device')
        if target is None and args:
            candidate = args[0]
            if isinstance(candidate, (str, torch.device, torch.Tensor)):
                target = candidate
        if isinstance(target, torch.Tensor):
            target = target.device
        return (target is not None and
                (str(target).lower() == 'gpu' or
                 str(target).lower().startswith('cuda')))

    def guarded_llm_to(self, *args, **kwargs):
        """Prevent an external driver from undoing CPU offload.

        cross_val.py historically does `model.LLM = model.LLM.to(llm_device)`
        even after LynxBase has decided the 7B model cannot fit on CUDA.
        LynxBase.to() cannot catch this direct submodule operation.
        """
        if (getattr(self, '_lynx_protect_cpu_offload', False) and
                _is_cuda_destination(args, kwargs)):
            if not getattr(self, '_lynx_cpu_offload_guard_logged', False):
                print('### Lynx CUDA guard: ignored model.LLM.to(cuda): this LLM '
                      'was deliberately offloaded to CPU to avoid CUDA OOM. '
                      'In cross_val.py, use model.llm_device rather than the '
                      'original --llm_device for a post-constructor .to().',
                      flush=True)
                self._lynx_cpu_offload_guard_logged = True
            return self
        return model_class.to(self, *args, **kwargs)

    def guarded_llm_cuda(self, device=None):
        """Protect direct model.LLM.cuda() as well as model.LLM.to()."""
        if getattr(self, '_lynx_protect_cpu_offload', False):
            if not getattr(self, '_lynx_cpu_offload_guard_logged', False):
                print('### Lynx CUDA guard: ignored model.LLM.cuda() because '
                      'the LLM is CPU-offloaded by the VRAM budget.', flush=True)
                self._lynx_cpu_offload_guard_logged = True
            return self
        return model_class.cuda(self, device=device)

    compatible_class = type(
        'LlamaForCausalLMWithGeneration', bases,
        {
            '__module__': __name__,
            '__doc__': 'Generation-compatible custom LLaMA with legacy KV caches and valid position IDs.',
            '_lynx_legacy_cache_wrapper': True,
            '_supports_default_dynamic_cache': classmethod(supports_modern_cache),
            'prepare_inputs_for_generation': prepare_legacy_inputs_for_generation,
            'forward': guarded_legacy_forward,
            'to': guarded_llm_to,
            'cuda': guarded_llm_cuda,
        },
    )
    globals()['LlamaForCausalLMWithGeneration'] = compatible_class
    return compatible_class



def _lynx_validate_logits(scores, *, step, policy='repair'):
    """Clean *partly* corrupt FP32 logits without hiding a broken model.

    A row of all-NaN/-Inf values has no probability distribution; failing with
    actionable diagnostics is safer than silently assigning random predictions
    in a scientific cross-validation run.
    """
    if policy not in ('repair', 'strict'):
        raise ValueError('invalid_logits_policy must be repair or strict')
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError(f'Expected [batch, vocab] logits, got {tuple(scores.shape)}')
    raw = scores.float()
    bad = torch.isnan(raw) | torch.isposinf(raw)
    if bool(bad.any()):
        if policy == 'strict':
            raise RuntimeError(
                f'Lynx numerical error: NaN/+Inf LLM logits at decode step {step}; '
                f'corrupted={int(bad.sum())}/{raw.numel()} (device={raw.device}, '
                f'dtype={scores.dtype}). Check training gradients and model precision.'
            )
        warnings.warn(
            f'Lynx numerical warning: replacing {int(bad.sum())} NaN/+Inf logits '
            f'at decode step {step}. These samples need validation.',
            RuntimeWarning, stacklevel=2,
        )
        positive = torch.isposinf(raw)
        raw = raw.masked_fill(torch.isnan(raw), float('-inf'))
        # Multiple +Inf logits have equal maximal probability. Preserve that
        # ordering instead of saturating them to an arbitrary finite value.
        inf_rows = positive.any(dim=-1, keepdim=True)
        raw = torch.where(inf_rows, torch.where(positive, torch.zeros_like(raw),
                                              torch.full_like(raw, float('-inf'))), raw)
    valid_rows = torch.isfinite(raw).any(dim=-1)
    if not bool(valid_rows.all()):
        rows = (~valid_rows).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(
            f'Lynx numerical error: ALL model logits are non-finite '
            f'at decode step {step}, batch rows={rows[:10]} '
            f'(device={scores.device}, dtype={scores.dtype}). '
            'This is not a sampling-filter issue. Check non-finite bridge '
            'activations/weights, unstable training loss and FP16 overflows. '
            'Try gpu_vision_dtype: float32, or run the vision encoder on CPU.'
        )
    return raw


def _lynx_repair_empty_candidates(processed, reference, *, step, eos_ids=(),
                                  min_length=0):
    """Recover when filtering, rather than the model, eliminates every token.

    Keep the highest-scoring *valid* reference token. Relax min_length only if
    unavoidable (e.g. the only viable token is EOS). Never fabricate logits.
    """
    missing = ~torch.isfinite(processed).any(dim=-1)
    if not bool(missing.any()):
        return processed
    restore = reference.clone()
    if step + 1 < int(min_length) and eos_ids:
        restore[:, list(eos_ids)] = float('-inf')
    restore_valid = torch.isfinite(restore).any(dim=-1)
    if not bool((restore_valid | ~missing).all()):
        # If EOS is the sole available valid token, stop early rather than
        # producing NaN probabilities or sampling an arbitrary token.
        restore = torch.where(restore_valid[:, None], restore, reference)
    if not bool((torch.isfinite(restore).any(dim=-1) | ~missing).all()):
        raise RuntimeError(f'Lynx numerical error: no finite logit at step {step}')
    chosen = restore.argmax(dim=-1)
    rows = missing.nonzero(as_tuple=True)[0]
    processed[rows] = float('-inf')
    processed[rows, chosen[rows]] = 0.0
    warnings.warn(
        f'Lynx generation filters removed every token for {len(rows)} batch '
        f'row(s) at step {step}; using the highest-scoring valid unfiltered token. '
        'Consider relaxing no_repeat_ngram_size or min_length.',
        RuntimeWarning, stacklevel=2,
    )
    return processed


def _lynx_generate_legacy_tokens(
    llm, inputs_embeds, attention_mask, *, do_sample=False, num_beams=1,
    max_new_tokens=64, min_length=2, top_p=0.9, top_k=3,
    repetition_penalty=1.0, no_repeat_ngram_size=3,
    length_penalty=1.0, num_return_sequences=1, temperature=1.0,
    pad_token_id=None, eos_token_id=None, invalid_logits_policy='repair',
):
    """Decode with the project's *tuple-based* LLaMA KV cache, not HF generate.

    The 2023 Lynx LLaMA expects position_ids to be [batch, current_tokens].
    New Transformers generation versions can supply stale full-prompt IDs on
    cached 1-token steps, accidentally broadening RoPE's batch dimension.
    By calling llm.forward explicitly, with current position IDs, we avoid
    the incompatible HF cache/position pipeline entirely. The prefill is run
    once; subsequent steps reuse the tuple KV cache, on CPU or CUDA.

    Supports greedy, temperature/top-k/top-p sampling, batch/beam expansion,
    repetition penalties, no-repeat n-grams, minimum length and EOS stopping.
    Beam sampling is an approximation to HF beam sampling. For model parity,
    keep do_sample=False when using num_beams > 1.
    """
    if inputs_embeds.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError('Expected inputs_embeds [batch, length, hidden] and attention_mask [batch, length]')
    batch, prefix_length, _ = inputs_embeds.shape
    if tuple(attention_mask.shape) != (batch, prefix_length):
        raise ValueError('Legacy decoder: mask does not match input embeddings')
    if not (1 <= num_beams and 1 <= num_return_sequences):
        raise ValueError('num_beams and num_return_sequences must be positive')
    if num_beams > 1 and num_return_sequences > num_beams:
        raise ValueError('num_return_sequences must be <= num_beams in beam mode')
    if max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    if repetition_penalty <= 0 or temperature <= 0:
        raise ValueError('repetition_penalty and temperature must be positive')
    if not (0 < top_p <= 1):
        raise ValueError('top_p must be in (0, 1]')

    target_device = inputs_embeds.device
    attention_mask = _lynx_normalize_attention_mask(
        attention_mask.to(device=target_device), name='decoder_attention_mask')
    positions = _lynx_positions_from_mask(
        attention_mask, current_length=prefix_length,
        name='decoder_attention_mask')
    # Crucial: [batch, prefix_length], never an unbatched [prefix_length].
    result = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=positions.contiguous(),
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    )
    scores = result.logits[:, -1, :].float()
    cache = result.past_key_values
    if not isinstance(cache, (tuple, list)) or not cache:
        raise RuntimeError('The legacy LLaMA must return a tuple-based KV cache')
    if any(not isinstance(layer, (tuple, list)) or len(layer) != 2
           or layer[0].ndim != 4 or layer[0].shape[0] != batch for layer in cache):
        raise RuntimeError('Unexpected legacy cache layout after prefill')

    vocab_size = scores.shape[-1]
    eos_ids = set()
    if eos_token_id is not None:
        eos_ids = ({int(eos_token_id)} if isinstance(eos_token_id, int)
                   else {int(x) for x in eos_token_id})
    fill_token = int(pad_token_id if pad_token_id is not None else
                     (min(eos_ids) if eos_ids else 0))
    beams = int(num_beams)
    # For sampling with a single beam, allow independent return sequences.
    paths = beams if beams > 1 else (int(num_return_sequences) if do_sample else 1)
    if paths > 1:
        cache = tuple(tuple(t.repeat_interleave(paths, dim=0) for t in layer)
                      for layer in cache)
        scores = scores.repeat_interleave(paths, dim=0)
        attention_mask = attention_mask.repeat_interleave(paths, dim=0)
    active_batch = batch * paths
    generated = torch.empty((active_batch, 0), dtype=torch.long, device=target_device)
    finished = torch.zeros(active_batch, dtype=torch.bool, device=target_device)
    beam_scores = torch.zeros((batch, paths), dtype=torch.float32, device=target_device)
    if beams > 1:
        beam_scores[:, 1:] = float('-inf')

    for step in range(int(max_new_tokens)):
        # Model-origin NaNs and filter-origin all-masked rows are different
        # failures: repair the latter, never generate from an all-NaN model.
        reference = _lynx_validate_logits(scores, step=step,
                                          policy=invalid_logits_policy)
        processed = reference.clone()
        if generated.size(1) and repetition_penalty != 1.0:
            # HF repetition penalty applies once per distinct observed token.
            history = torch.zeros_like(processed, dtype=torch.bool)
            history.scatter_(1, generated, True)
            penalty_scores = torch.where(processed < 0,
                                         processed * repetition_penalty,
                                         processed / repetition_penalty)
            processed = torch.where(history, penalty_scores, processed)
        ngram = int(no_repeat_ngram_size)
        if ngram > 0 and generated.shape[1] >= ngram - 1:
            # Generated text is all the token information available here:
            # the multimodal prompt is provided as embeddings, not token IDs.
            for row in range(active_batch):
                previous = generated[row].tolist()
                if ngram == 1:
                    banned = set(previous)
                else:
                    context = tuple(previous[-(ngram - 1):])
                    banned = {
                        previous[i + ngram - 1]
                        for i in range(len(previous) - ngram + 1)
                        if tuple(previous[i:i + ngram - 1]) == context
                    }
                if banned:
                    processed[row, list(banned)] = float('-inf')
        if step + 1 < int(min_length) and eos_ids:
            processed[:, list(eos_ids)] = float('-inf')
        # Stabilize before temperature scaling: subtracting the per-row max
        # cannot change sampling probabilities but prevents +Inf overflow when
        # temperature is small. Verify at least one finite candidate first.
        processed = _lynx_repair_empty_candidates(
            processed, reference, step=step, eos_ids=eos_ids, min_length=min_length)
        if do_sample:
            max_scores = processed.max(dim=-1, keepdim=True).values
            processed = (processed - max_scores) / float(temperature)
            if int(top_k) > 0 and int(top_k) < vocab_size:
                cutoff = processed.topk(int(top_k), dim=-1).values[:, -1, None]
                processed.masked_fill_(processed < cutoff, float('-inf'))
            if top_p < 1:
                ordered, indexes = torch.sort(processed, dim=-1, descending=True)
                ordered_probs = torch.softmax(ordered, dim=-1)
                cumulative = ordered_probs.cumsum(dim=-1)
                # Keep the first token that brings cumulative mass over top_p.
                remove = (cumulative - ordered_probs) >= top_p
                remove[:, 0] = False
                processed.scatter_(1, indexes,
                                   ordered.masked_fill(remove, float('-inf')))
            processed = _lynx_repair_empty_candidates(
                processed, reference, step=step, eos_ids=eos_ids, min_length=min_length)
        # A previously finished path can only emit PAD without changing score.
        if finished.any():
            processed[finished] = float('-inf')
            processed[finished, fill_token] = 0.0
        if beams > 1:
            log_probs = torch.log_softmax(processed, dim=-1)
            combined = log_probs + beam_scores.reshape(-1, 1)
            # Finished beams must not accrue a penalty for PAD continuation.
            if finished.any():
                combined[finished] = float('-inf')
                combined[finished, fill_token] = beam_scores.reshape(-1)[finished]
            candidate_scores = combined.view(batch, paths * vocab_size)
            if do_sample:
                # Beam sampling approximation: sample candidate paths by mass,
                # then retain the highest-scoring sampled expansions.
                # Scores for disabled beams may be -Inf; at least one active
                # beam must have a finite score. Softmax in FP32 is stable.
                if not bool(torch.isfinite(candidate_scores).any(dim=-1).all()):
                    raise RuntimeError(
                        f'Lynx numerical error: no valid beam candidates at step {step}'
                    )
                beam_probs = torch.softmax(candidate_scores.float(), dim=-1)
                candidates = torch.multinomial(
                    beam_probs, num_samples=paths, replacement=True,
                )
                selected_scores = candidate_scores.gather(1, candidates)
                ordered = selected_scores.argsort(dim=1, descending=True)
                selected = candidates.gather(1, ordered)
                selected_scores = selected_scores.gather(1, ordered)
            else:
                selected_scores, selected = candidate_scores.topk(paths, dim=1)
            previous_beams = torch.div(selected, vocab_size, rounding_mode='floor')
            new_tokens = (selected % vocab_size).reshape(-1)
            parent_indices = (previous_beams +
                              torch.arange(batch, device=target_device)[:, None] * paths).reshape(-1)
            cache = tuple(tuple(t.index_select(0, parent_indices) for t in layer)
                          for layer in cache)
            attention_mask = attention_mask.index_select(0, parent_indices)
            generated = generated.index_select(0, parent_indices)
            finished = finished.index_select(0, parent_indices)
            beam_scores = selected_scores
        else:
            if do_sample:
                probs = torch.softmax(processed.float(), dim=-1)
                # No NaNs can reach multinomial: a non-finite distribution
                # is reconstructed from an original finite logit, or raises
                # a precise numerical error if the model itself is broken.
                valid_probs = (torch.isfinite(probs).all(dim=-1) &
                               (probs >= 0).all(dim=-1) &
                               (probs.sum(dim=-1) > 0))
                if not bool(valid_probs.all()):
                    rows = (~valid_probs).nonzero(as_tuple=True)[0]
                    repaired = _lynx_repair_empty_candidates(
                        torch.full_like(reference, float('-inf')), reference,
                        step=step, eos_ids=eos_ids, min_length=min_length)
                    choices = repaired.argmax(dim=-1)
                    probs[rows] = 0.0
                    probs[rows, choices[rows]] = 1.0
                    warnings.warn(
                        f'Lynx repaired {len(rows)} invalid sampling distribution(s) '
                        f'at step {step} using finite unfiltered logits.',
                        RuntimeWarning, stacklevel=2,
                    )
                new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                new_tokens = processed.argmax(dim=-1)
        generated = torch.cat((generated, new_tokens[:, None]), dim=1)
        if eos_ids:
            for stop_token in eos_ids:
                finished |= new_tokens == stop_token
        if step + 1 == max_new_tokens or bool(torch.all(finished)):
            break

        # Cache only contains earlier tokens; this single-token forward adds
        # the newly selected token to it. This bypasses all HF generation code.
        attention_mask = torch.cat((attention_mask, torch.ones(
            (active_batch, 1), dtype=torch.long, device=target_device)), dim=1)
        past_length = cache[0][0].size(2)
        new_positions = _lynx_positions_from_mask(
            attention_mask, current_length=1, past_length=past_length,
            name='decoder_attention_mask')
        assert tuple(new_positions.shape) == (active_batch, 1)
        if any(layer[0].shape[0] != active_batch for layer in cache):
            raise RuntimeError('KV cache batch dimension changed unexpectedly')
        result = llm(
            input_ids=new_tokens[:, None],
            attention_mask=attention_mask,
            position_ids=new_positions.contiguous(),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = result.past_key_values
        scores = result.logits[:, -1, :].float()

    if beams > 1:
        lengths = torch.full((batch, paths), generated.shape[1], device=target_device,
                             dtype=torch.float32)
        if eos_ids:
            for i in range(generated.shape[1]):
                is_eos = torch.zeros_like(finished)
                for stop_token in eos_ids:
                    is_eos |= generated[:, i] == stop_token
                # Only use first EOS occurrence to calculate sequence length.
                lengths.view(-1)[(lengths.view(-1) == generated.shape[1]) & is_eos] = i + 1
        normalized = beam_scores / lengths.clamp(min=1).pow(float(length_penalty))
        selected_beams = normalized.topk(int(num_return_sequences), dim=1).indices
        selected_rows = (selected_beams +
                         torch.arange(batch, device=target_device)[:, None] * paths).reshape(-1)
        generated = generated.index_select(0, selected_rows)

    # Trim EOS/PAD continuations; keep rectangular output for batch_decode.
    if eos_ids:
        for i in range(generated.shape[0]):
            seq = generated[i]
            for pos in range(seq.numel()):
                if int(seq[pos]) in eos_ids:
                    if pos + 1 < seq.numel():
                        generated[i, pos + 1:] = fill_token
                    break
    return generated


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


#allgather = AllGather.apply


def disable_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def _lynx_enable_decoder_activation_checkpointing(llm):
    """Checkpoint legacy LLaMA decoder layers without changing state_dict keys.

    This is especially important for a *frozen* CPU LLM receiving embeddings
    from a trainable GPU bridge: the LLM still needs a backward graph for the
    bridge gradient. Do not wrap it in torch.no_grad(). Uses non-reentrant
    checkpointing so tensor / tuple outputs and kwargs are supported.
    """
    layers = getattr(getattr(llm, 'model', None), 'layers', ())
    enabled = 0
    for layer in layers:
        if getattr(layer, '_lynx_ram_checkpointed', False):
            continue
        original_forward = layer.forward

        def memory_efficient_forward(*args, __forward=original_forward, **kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if (torch.is_grad_enabled() and isinstance(x, torch.Tensor)
                    and x.requires_grad and not kwargs.get('use_cache', False)):
                return _torch_activation_checkpoint(
                    __forward, *args, use_reentrant=False,
                    preserve_rng_state=True, **kwargs)
            return __forward(*args, **kwargs)

        layer.forward = memory_efficient_forward
        layer._lynx_ram_checkpointed = True
        enabled += 1
    return enabled


class LynxBase(nn.Module):
    """Lynx with device-aware CPU/GPU startup and memory-efficient weight IO.

    Runtime --device and --llm_device are read from sys.argv if the driver
    does not pass them via config or explicit constructor kwargs. Both gpu
    values are mapped to CUDA; no user CLI changes are necessary.
    """
    def __init__(self, config=None, freeze_vit=True, freeze_llm=True,
                 load_bridge=False, device=None, llm_device=None):
        """Build on CPU first, then place each component on its requested device.

        Device precedence: explicit constructor arguments > cross_val.py CLI
        (--device/--llm_device) > config dictionary > CPU defaults.
        The command-line values 'gpu' and 'cuda' both mean CUDA. The CPU and
        GPU loader paths are selected independently for vision and language.

        Existing cross_val.py invocations need not change:
          --device cpu --llm_device cpu
          --device gpu --llm_device gpu
        Mixed CPU/GPU placement also works (when CUDA is available).
        """
        super().__init__()
        if config is None:
            raise ValueError('LynxBase requires a model config dictionary')
        self.vision_device, self.llm_device = self.resolve_runtime_devices(
            config, device=device, llm_device=llm_device,
        )
        print(f'### Lynx runtime: vision={self.vision_device}, '
              f'LLM={self.llm_device}', flush=True)

        self._requested_vision_device = self.vision_device
        self._requested_llm_device = self.llm_device
        self._lynx_auto_offload = False
        self._lynx_warned_external_to = False
        self.freeze_vit = freeze_vit
        self.freeze_llm = freeze_llm
        self.init_params = []
        self._lynx_crossval_activation_checkpointing = bool(
            config.get('crossval_gradient_checkpointing', True))

        # Construct and load weights on the CPU to avoid temporary CUDA copies
        # and to support sequential memory-mapped checkpoint loading.
        self.vision_encoder, missing_keys = self.build_vision_encoder(config, freeze_params=freeze_vit)
        self.update_init_params([f'vision_encoder.{k}' for k in missing_keys])

        self.tokenizer, num_new_tokens = build_tokenizer(config['LLM'], use_left_pad=config['use_left_pad'])
        # Some legacy additional.py versions report a new pad token but
        # leave pad_token_id=-1 and len(tokenizer) unchanged. Repair
        # padding before LLaMA config/model construction.
        _ensure_valid_tokenizer_pad(self.tokenizer)
        llm_on_cpu = self.llm_device.type == 'cpu'
        sequential_llm = (config.get('cpu_sequential_llm', True) if llm_on_cpu
                          else config.get('gpu_sequential_llm', True))
        if sequential_llm:
            self.LLM, missing_keys = self._build_llm(
                config, freeze_params=freeze_llm,
                num_new_tokens=num_new_tokens, use_mmap=False,
                sequential=True,
            )
        else:
            mmap_llm = (config.get('mmap_llm', False) if llm_on_cpu
                        else config.get('gpu_mmap_llm', config.get('mmap_llm', False)))
            if mmap_llm:
                self.LLM, missing_keys = self.build_LLM_mmap(
                    config, freeze_params=freeze_llm,
                    num_new_tokens=num_new_tokens,
                )
            else:
                self.LLM, missing_keys = self.build_LLM(
                    config, freeze_params=freeze_llm,
                    num_new_tokens=num_new_tokens,
                )
        self.update_init_params([f'LLM.{k}' for k in missing_keys])

        # Bridge: Vision2Text
        self.bridge, missing_keys = self.build_bridge(config, load_params=load_bridge)
        self.update_init_params([f'bridge.{k}' for k in missing_keys])

        # Video
        self.video_encoding = config.get("video_encoding", "")
        if self.video_encoding:
            self.add_frame_pos = config['add_frame_pos']
            if self.add_frame_pos:
                self.absolute_frame_pos_embed = nn.Parameter(
                    torch.zeros(1, config['data']['num_frames'], 1, self.vision_width))
                trunc_normal_(self.absolute_frame_pos_embed, std=.02)
                self.update_init_params(['absolute_frame_pos_embed'])

            elif self.video_encoding == 'concate':
                # concate all video frames
                pass

            else:
                raise NotImplementedError(f"video_encoding == {self.video_encoding}")

        if os.path.exists(config.get('checkpoint', '')):
            self.load_pretrained(config['checkpoint'], config)

        # The 2023 model stores RoPE inverse frequencies in *buffers*, not
        # parameters. Check them after loading both checkpoints, while on CPU.
        if not os.path.exists(config.get('checkpoint', '')):
            self._lynx_rotary_report = _lynx_validate_llama_rotary_buffers(
                self.LLM, repair=config.get('legacy_llama_repair_rope_buffers', True))

        # Transfer exactly once after all CPU-side checkpoint reads finish.
        self.place_model_on_runtime_devices(config)

    @staticmethod
    def _get_cli_value(option, argv=None):
        """Read one known flag without consuming or changing cross_val.py args."""
        args = sys.argv[1:] if argv is None else list(argv)
        flag = f'--{option}'
        for i, item in enumerate(args):
            if item == flag:
                if i + 1 >= len(args) or args[i + 1].startswith('--'):
                    raise ValueError(f'{flag} requires a value')
                return args[i + 1]
            if item.startswith(flag + '='):
                return item[len(flag) + 1:]
        return None

    @staticmethod
    def _normalize_device(value, name):
        """Accept the project's 'gpu' alias as well as native torch devices."""
        if isinstance(value, torch.device):
            dev = value
        else:
            if not isinstance(value, str):
                raise ValueError(f'{name} must be cpu, gpu, cuda, or cuda:N: {value!r}')
            raw = value.strip().lower()
            if raw == 'gpu':
                raw = 'cuda'
            try:
                dev = torch.device(raw)
            except (TypeError, RuntimeError, ValueError) as exc:
                raise ValueError(f'Invalid {name}: {value!r}') from exc
        if dev.type not in ('cpu', 'cuda'):
            raise ValueError(f'{name} only supports CPU or CUDA, not {dev!s}')
        if dev.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f'{name}={value!s} requests GPU but PyTorch CUDA is unavailable'
                )
            if dev.index is not None and dev.index >= torch.cuda.device_count():
                raise ValueError(f'{name} CUDA index out of range: {dev.index}')
        return dev

    @classmethod
    def resolve_runtime_devices(cls, config, *, device=None, llm_device=None,
                                argv=None):
        """Explicit kwargs override CLI, and CLI overrides CROSSVAL.yaml."""
        vision_setting = device
        if vision_setting is None:
            vision_setting = cls._get_cli_value('device', argv=argv)
        if vision_setting is None:
            vision_setting = config.get('device', 'cpu')

        llm_setting = llm_device
        if llm_setting is None:
            llm_setting = cls._get_cli_value('llm_device', argv=argv)
        if llm_setting is None:
            llm_setting = config.get('llm_device', vision_setting)

        return (cls._normalize_device(vision_setting, 'device'),
                cls._normalize_device(llm_setting, 'llm_device'))

    @staticmethod
    def _cuda_dtype(value, target, label):
        """Choose a low-memory CUDA dtype before any GPU allocation.

        Old configurations that used 'auto' previously retained float32.
        In GPU mode 'auto' now means FP16, or BF16 when supported.
        """
        if target.type != 'cuda':
            return None
        name = str(value or 'float16').lower()
        if name == 'auto':
            name = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        mapping = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
                   'float32': torch.float32}
        if name not in mapping:
            raise ValueError(f'Invalid {label}: {value!r}')
        if name == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ValueError(f'{label}=bfloat16 requires a BF16-capable GPU')
        return mapping[name]

    @staticmethod
    def _module_weight_bytes(module, target_dtype=None):
        """Count live model tensors, not temporary state dictionaries.

        Default named_parameters removes duplicate Parameter objects; storage
        aliases represented by different Parameter objects can still cause a
        conservative over-estimate, which is safe for a VRAM budget.
        """
        num_bytes = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            size = (torch.empty((), dtype=target_dtype).element_size()
                    if target_dtype is not None and tensor.is_floating_point()
                    else tensor.element_size())
            num_bytes += tensor.numel() * size
        return num_bytes

    @staticmethod
    def _choose_gpu_plan(available, reserve, vision_bytes, llm_bytes,
                         want_vision, want_llm, policy='auto'):
        """Pure sizing decision; callers pass actual free VRAM (not total VRAM).

        Results are (vision_on_gpu, llm_on_gpu, reason). In auto mode prefer
        retaining the more compute-heavy LLM on CUDA when it fits alone.
        No amount of empty_cache() can make two models fit if their weights
        exceed free VRAM, so never attempt the doomed full-model transfer.
        """
        if policy not in ('auto', 'strict'):
            raise ValueError('gpu_memory_policy must be auto or strict')
        required = ((vision_bytes if want_vision else 0) +
                    (llm_bytes if want_llm else 0))
        budget = max(0, int(available) - int(reserve))
        if required <= budget:
            return want_vision, want_llm, 'requested placement fits weight budget'
        if policy == 'strict':
            raise RuntimeError(
                'CUDA weight budget exceeded before model transfer: '
                f'weights={required / 1024**3:.2f} GiB, '
                f'free={available / 1024**3:.2f} GiB, '
                f'reserved-for-runtime={reserve / 1024**3:.2f} GiB. '
                'Use gpu_memory_policy: auto for CPU offload, a larger GPU, '
                'or a model with lower VRAM requirements.'
            )
        if want_llm and llm_bytes <= budget:
            return False, True, 'offload vision encoder and bridge to CPU'
        if want_vision and vision_bytes <= budget:
            return True, False, 'offload LLM to CPU'
        return False, False, 'GPU cannot hold either component with the configured reserve'

    def _set_frame_embedding_device(self, device, dtype=None):
        if not hasattr(self, 'absolute_frame_pos_embed'):
            return
        param = self.absolute_frame_pos_embed
        new = param.detach().to(device=device, dtype=dtype or param.dtype)
        # Replace the Parameter only if placement/dtype changed.
        if new is not param:
            self.absolute_frame_pos_embed = nn.Parameter(
                new, requires_grad=param.requires_grad,
            )

    def place_model_on_runtime_devices(self, config=None):
        """Memory-budgeted CUDA placement with FP16 transfer in one operation.

        A 7B LLM plus a 1B ViT cannot fit on a 15GB GPU at FP32 and may not
        fit together even in FP16. Never do `self.to('cuda')` in that case.
        Mixed CPU/GPU inference is supported by generate(). CPU-mode startup
        preserves the original float32 behaviour.

        Optional YAML controls:
          gpu_memory_policy: auto (default; CPU offload when needed) | strict
          gpu_memory_reserve_gib: 2.0 (room for KV cache/activations/kernels)
          gpu_llm_dtype: float16 (default) | bfloat16 | float32 | auto
          gpu_vision_dtype: float16 (default) | bfloat16 | float32 | auto
          gpu_bridge_dtype: gpu_vision_dtype (default) | float32 for training
          cpu_offload_dtype: float32 (safe) | bfloat16 | float16 (less host RAM)
        """
        config = config or {}
        requested_vision = self._requested_vision_device
        requested_llm = self._requested_llm_device
        vision_target, llm_target = requested_vision, requested_llm
        vision_dtype = self._cuda_dtype(config.get('gpu_vision_dtype', 'float16'),
                                       requested_vision, 'gpu_vision_dtype')
        # Keep the trainable bridge in FP32 during cross-validation unless the
        # caller explicitly requests another dtype. Unlike frozen ViT weights,
        # FP16 trainable parameters can underflow without mixed-precision training.
        bridge_dtype = self._cuda_dtype(
            config.get('gpu_bridge_dtype', config.get('gpu_vision_dtype', 'float16')),
            requested_vision, 'gpu_bridge_dtype'
        )
        llm_dtype = self._cuda_dtype(config.get('gpu_llm_dtype', 'float16'),
                                    requested_llm, 'gpu_llm_dtype')
        have_cuda = requested_vision.type == 'cuda' or requested_llm.type == 'cuda'

        # Make room for runtime allocations, especially beam-search KV caches.
        if have_cuda:
            device = requested_llm if requested_llm.type == 'cuda' else requested_vision
            free, total = torch.cuda.mem_get_info(device)
            reserve_gib = float(config.get('gpu_memory_reserve_gib', 2.0))
            if reserve_gib < 0:
                raise ValueError('gpu_memory_reserve_gib must be nonnegative')
            reserve = int(reserve_gib * (1024 ** 3))
            vision_bytes = (self._module_weight_bytes(self.vision_encoder, vision_dtype) +
                            self._module_weight_bytes(self.bridge, bridge_dtype))
            if hasattr(self, 'absolute_frame_pos_embed'):
                p = self.absolute_frame_pos_embed
                vision_bytes += p.numel() * (2 if vision_dtype in (torch.float16, torch.bfloat16)
                                               else p.element_size())
            llm_bytes = self._module_weight_bytes(self.LLM, llm_dtype)
            v, l, reason = self._choose_gpu_plan(
                free, reserve, vision_bytes, llm_bytes,
                requested_vision.type == 'cuda', requested_llm.type == 'cuda',
                policy=config.get('gpu_memory_policy', 'auto'),
            )
            vision_target = requested_vision if v else torch.device('cpu')
            llm_target = requested_llm if l else torch.device('cpu')
            self._lynx_auto_offload = bool(
                (requested_vision.type == 'cuda' and not v) or
                (requested_llm.type == 'cuda' and not l)
            )
            print(f'### Lynx CUDA: free={free / 1024**3:.2f} GiB '
                  f'of {total / 1024**3:.2f} GiB, '
                  f'vision+bridge={vision_bytes / 1024**3:.2f} GiB, '
                  f'LLM={llm_bytes / 1024**3:.2f} GiB, '
                  f'headroom={reserve_gib:.2f} GiB', flush=True)
            print(f'### Lynx CUDA plan: vision={vision_target}, LLM={llm_target}; '
                  f'{reason}', flush=True)

        # Move the largest component first; cast FP32 CPU weights to FP16/BF16
        # during CPU->CUDA transfer, NOT after placing FP32 tensors on CUDA.
        policy = config.get('gpu_memory_policy', 'auto')
        if llm_target.type == 'cuda':
            try:
                self.LLM.to(device=llm_target, dtype=llm_dtype)
            except torch.OutOfMemoryError as exc:
                if policy == 'strict':
                    raise
                # Some custom CUDA kernels/allocators use more memory than the
                # estimated weights. Recover gracefully instead of crashing.
                print('### Lynx: CUDA OOM transferring LLM; offloading LLM to CPU',
                      flush=True)
                self.LLM.to('cpu')
                llm_target = torch.device('cpu')
                self._lynx_auto_offload = True
                torch.cuda.empty_cache()
                if requested_vision.type == 'cuda':
                    vision_target = requested_vision
        if llm_target.type == 'cpu':
            # Default to float32 for compatibility with legacy CPU ops. When
            # CUDA was requested but the LLM was offloaded, float16/bfloat16
            # can optionally reduce HOST RAM by about 2x. Test CPU operators
            # and output quality before choosing this memory-saving option.
            offloaded = requested_llm.type == 'cuda'
            cpu_dtype_name = (config.get('cpu_offload_dtype', 'float32')
                              if offloaded else 'float32')
            cpu_dtypes = {'float32': torch.float32, 'float16': torch.float16,
                          'bfloat16': torch.bfloat16}
            if cpu_dtype_name not in cpu_dtypes:
                raise ValueError('cpu_offload_dtype must be float32, float16, or bfloat16')
            if offloaded and cpu_dtype_name != 'float32':
                print(f'### Lynx: CPU-offloaded LLM using {cpu_dtype_name}; '
                      'lower host RAM usage, but verify legacy CPU operator support',
                      flush=True)
            self.LLM.to(device='cpu', dtype=cpu_dtypes[cpu_dtype_name])

        if vision_target.type == 'cuda':
            try:
                self.vision_encoder.to(device=vision_target, dtype=vision_dtype)
                self.bridge.to(device=vision_target, dtype=bridge_dtype)
                self._set_frame_embedding_device(vision_target, vision_dtype)
            except torch.OutOfMemoryError:
                if policy == 'strict':
                    raise
                print('### Lynx: CUDA OOM transferring vision/bridge; '
                      'offloading them to CPU', flush=True)
                self.vision_encoder.to('cpu')
                self.bridge.to('cpu')
                self._set_frame_embedding_device(torch.device('cpu'))
                vision_target = torch.device('cpu')
                self._lynx_auto_offload = True
                torch.cuda.empty_cache()
        else:
            self.vision_encoder.to('cpu')
            self.bridge.to('cpu')
            self._set_frame_embedding_device(torch.device('cpu'))

        self.vision_device, self.llm_device = vision_target, llm_target
        # Lock only submodule transfers back to CUDA when the budget selected
        # CPU offload. This catches cross_val.py line 219 without monkeypatching
        # PyTorch globally or blocking dtype-only / CPU module transfers.
        self.LLM._lynx_protect_cpu_offload = bool(
            requested_llm.type == 'cuda' and llm_target.type == 'cpu'
        )
        if self.LLM._lynx_protect_cpu_offload:
            print('### Lynx: CPU LLM offload protected against external '
                  'model.LLM.to(cuda) calls', flush=True)
        if self.llm_device.type == 'cpu' and hasattr(self.LLM, 'config'):
            # The custom LLaMA reads this config in its forward method.
            self.LLM.config.use_flash_attn = False
            if hasattr(self.LLM, 'model') and hasattr(self.LLM.model, 'config'):
                self.LLM.model.config.use_flash_attn = False
        print(f'### Lynx CUDA patch: {LYNX_CUDA_MEMORY_PATCH_VERSION}; '
              f'vision/bridge={self.vision_device}, LLM={self.llm_device}', flush=True)
        if have_cuda:
            try:
                free_now, _ = torch.cuda.mem_get_info(device)
                allocated = torch.cuda.memory_allocated(device)
                print(f'### Lynx CUDA after placement: '
                      f'free={free_now / 1024**3:.2f} GiB, '
                      f'allocated_by_torch={allocated / 1024**3:.2f} GiB', flush=True)
            except RuntimeError:
                pass
        return self

    def to(self, *args, **kwargs):
        """Protect a memory-budgeted split from cross_val.py model.to('cuda').

        A later full-model CUDA transfer would undo CPU offloading. Users can
        opt out through `model._lynx_auto_offload = False` if desired.
        """
        if getattr(self, '_lynx_auto_offload', False):
            target = kwargs.get('device', args[0] if args else None)
            if isinstance(target, (torch.device, str)):
                text = str(target).lower()
                if text == 'gpu' or text.startswith('cuda'):
                    if not self._lynx_warned_external_to:
                        print('### Lynx: ignoring whole-model .to(cuda) because '
                              'automatic CPU offload is active; use '
                              'place_model_on_runtime_devices() to adjust placement',
                              flush=True)
                        self._lynx_warned_external_to = True
                    return self
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        """Avoid undoing automatic CPU offload via a later model.cuda()."""
        if getattr(self, '_lynx_auto_offload', False):
            if not self._lynx_warned_external_to:
                print('### Lynx: ignoring whole-model .cuda() because '
                      'automatic CPU offload is active', flush=True)
                self._lynx_warned_external_to = True
            return self
        return super().cuda(device=device)

    def build_vision_encoder(self, config, freeze_params=True):
        """
        Args:
            load_params: False when building fine-tuning models
        """
        print(f"### Building ViT (Freeze: {freeze_params})", flush=True)

        if config['vision_encoder'] == 'eva_vit_1b':
            from models.vits.eva_vit import create_eva_vit_g
            model, missing_keys = create_eva_vit_g(config['image_res'], config.get('drop_path_rate', 0.0),
                                                   load_params=True)
            # set attrs
            self.vision_width = model.embed_dim

        else:
            raise NotImplementedError("Vision Encoder: ", config['vision_encoder'])

        if freeze_params:

            assert len(missing_keys) == 0

            for name, param in model.named_parameters():
                param.requires_grad = False

            print("### Desabling training and setting for evaluation")
            model = model.eval()
            model.train = disable_train

        return model, missing_keys

    def build_LLM(self, config, freeze_params=True, num_new_tokens=0):
        """Load the language model through the standard checkpoint reader."""
        return self._build_llm(config, freeze_params, num_new_tokens, use_mmap=False)

    def build_LLM_mmap(self, config, freeze_params=True, num_new_tokens=0):
        """Load the language model with the project's mmap sharded-weight reader."""
        return self._build_llm(config, freeze_params, num_new_tokens, use_mmap=True)

    def _finetuned_checkpoint_covers_llm(self, checkpoint_path, llm):
        """Metadata-only test: can the final checkpoint replace base Vicuna?

        Requires every non-adapter LLM parameter, with matching shapes. We do
        not skip base loading for adapter-only or partially matched checkpoints.
        This avoids reading the ~25 GiB base model unnecessarily when the
        project's final checkpoint already contains the full language model.
        """
        source = self._resolve_cpu_weights(checkpoint_path)
        if source is None:
            return False
        required = {f'LLM.{name}': tuple(param.shape)
                    for name, param in llm.named_parameters()
                    if 'adapter' not in name.lower()}
        if not required:
            return False
        paths, weight_map = source
        if weight_map is not None:
            keys = {k.removeprefix('model.') for k in weight_map}
            # For indexed checkpoints, tensor shapes will still be checked
            # during the full sequential load.
            covered = sum(name in keys for name in required)
        else:
            # mmap=True reads metadata but doesn't copy all tensor payloads.
            if len(paths) != 1:
                return False
            path = paths[0]
            if path.suffix == '.safetensors':
                from safetensors import safe_open
                with safe_open(str(path), framework='pt', device='cpu') as f:
                    shapes = {k.removeprefix('model.'): tuple(f.get_slice(k).get_shape())
                              for k in f.keys()}
            else:
                checkpoint = self._read_checkpoint(str(path), mmap=True,
                                                   weights_only=True)
                weights = checkpoint.get('model', checkpoint)
                if not isinstance(weights, dict):
                    return False
                shapes = {k.removeprefix('model.'): tuple(t.shape)
                          for k, t in weights.items() if isinstance(t, torch.Tensor)}
                del weights, checkpoint
            covered = sum(shapes.get(name) == shape for name, shape in required.items())
        print(f'### Lynx final-checkpoint LLM base coverage: '
              f'{covered}/{len(required)} parameter tensors', flush=True)
        return covered == len(required)

    def _build_llm(self, config, freeze_params=True, num_new_tokens=0,
                   use_mmap=False, sequential=None):
        """Shared loader; avoids maintaining two different initialization paths.

        Optional configuration:
          mmap_llm: use the project's existing sharded mmap reader.
          llm_low_cpu_mem_usage: use the Transformers/Accelerate low-RAM loader.
          llm_dtype: 'float16', 'bfloat16', 'float32', or 'auto'.
        The latter two are opt-in for compatibility with the project's
        customized (older) LLaMA implementation and hardware.
        """
        print(f"### Building LLM (Freeze: {freeze_params}, mmap: {use_mmap})", flush=True)
        self.use_dec_only = True
        self.use_adapter = config.get('use_adapter', False)

        if config['LLM'] != 'vicuna-7b':
            raise NotImplementedError("LLM: ", config['LLM'])

        from models.llms.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
        # Fix for Transformers >= 4.50 / 5.x: the custom Lynx LLaMA model
        # needs the entire GenerationMixin interface, not just .generate().
        LlamaForCausalLM = _with_generation_mixin(LlamaForCausalLM)
        print(f'### Lynx generation patch: {LYNX_GENERATION_PATCH_VERSION}', flush=True)

        rpath = os.path.join('data', config['LLM'])
        if not os.path.isdir(rpath):
            raise FileNotFoundError(f"LLM directory not found: {rpath}")

        # Sanitize the LLaMA config *before* constructing a Transformers
        # PretrainedConfig. Older Vicuna config.json files often declare
        # pad_token_id=-1, which recent Transformers warns about immediately
        # during config creation (not just during from_pretrained()).
        config_path = os.path.join(rpath, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as fp:
            raw_config = json.load(fp)
        if not isinstance(raw_config, dict):
            raise ValueError(f'Expected a JSON model configuration object: {config_path}')
        base_vocab = int(raw_config.get('vocab_size', len(self.tokenizer)))
        pad_id = _ensure_valid_tokenizer_pad(self.tokenizer)
        config_pad = pad_id if _valid_vocab_id(pad_id, base_vocab) else None
        if config_pad is None:
            for special in ('eos_token_id', 'unk_token_id', 'bos_token_id'):
                candidate = getattr(self.tokenizer, special, None)
                if _valid_vocab_id(candidate, base_vocab):
                    config_pad = candidate
                    break
        # None is an allowed config value if the tokenizer temporarily refers
        # to a newly added token beyond the *base* vocabulary. Its final ID
        # is set after resize_token_embeddings().
        raw_config['pad_token_id'] = config_pad
        for token_name in ('eos_token_id', 'bos_token_id'):
            candidate = getattr(self.tokenizer, token_name, None)
            if _valid_vocab_id(candidate, base_vocab):
                raw_config[token_name] = candidate
        if hasattr(LlamaConfig, 'from_dict'):
            text_config = LlamaConfig.from_dict(raw_config)
        else:
            # Compatibility with non-HF custom LlamaConfig classes.
            text_config = LlamaConfig.from_json_file(config_path)
        text_config.pad_token_id = config_pad
        for token_name in ('eos_token_id', 'bos_token_id'):
            if token_name in raw_config:
                setattr(text_config, token_name, raw_config[token_name])
        # FlashAttention needs a CUDA language model. Do not accidentally
        # enable a GPU-only kernel during a CPU run using the same YAML.
        text_config.use_flash_attn = (
            config.get('use_flash_attn', False)
            and getattr(self, 'llm_device', torch.device('cpu')).type == 'cuda'
        )
        text_config.use_adapter = self.use_adapter
        text_config.adapter_freq = config.get('adapter_freq', -1)
        text_config.freeze_params = freeze_params
        text_config.label_smoothing = config.get('label_smoothing', 0.0)

        # The explicit sequential-CPU path opens only one checkpoint shard at
        # a time, and never assembles a complete state_dict in host RAM.
        llm_device = getattr(self, 'llm_device', torch.device('cpu'))
        if sequential is None:
            # Preserve direct _build_llm(...) calls from existing project code.
            sequential = (config.get('cpu_sequential_llm', False)
                          if llm_device.type == 'cpu'
                          else config.get('gpu_sequential_llm', True))
        source = self._resolve_cpu_weights(rpath) if sequential else None
        if sequential and source is not None:
            dtype_setting = (config.get('cpu_llm_dtype', config.get('llm_dtype'))
                             if llm_device.type == 'cpu'
                             else config.get('gpu_llm_dtype', config.get('llm_dtype')))
            if llm_device.type == 'cpu' and dtype_setting not in (None, 'float32', 'auto'):
                raise ValueError('Sequential CPU LLM constructor uses float32; '
                                 'disable cpu_sequential_llm for another dtype')
            print(f'### Sequential LLM loading for {llm_device.type} '
                  f'(CPU checkpoint staging, one shard at a time)', flush=True)
            # Unlike from_pretrained(..., state_dict=...), this never forms a
            # merged multi-shard state dictionary. The model constructor is
            # deliberately retained to initialize missing adapters correctly.
            model = LlamaForCausalLM(text_config)
            final_path = config.get('checkpoint', '')
            skip_base = (freeze_params and
                         config.get('skip_base_llm_when_full_finetune', True) and
                         final_path and os.path.exists(final_path) and
                         self._finetuned_checkpoint_covers_llm(final_path, model))
            self._lynx_skipped_base_llm_weights = bool(skip_base)
            if skip_base:
                print('### Lynx RAM: full fine-tune checkpoint covers the LLM; '
                      'skip redundant base Vicuna weights', flush=True)
                report = {'missing': [k for k, _ in model.named_parameters()
                                      if 'adapter' in k], 'unexpected': []}
            else:
                report = self._load_cpu_weights(
                    model, source, normalize_prefix=False,
                    mmap=config.get('checkpoint_mmap', True),
                    weights_only=config.get('checkpoint_weights_only', True),
                    rebind=config.get('llm_zero_copy_load', bool(freeze_params)),
                    interpolate_vision=False,
                )
            # LLaMA adapters are intentionally absent from vanilla Vicuna
            # checkpoints. Other missing keys may be from custom model code.
            not_loaded = [] if skip_base else [k for k in report['missing'] if 'adapter' not in k]
            if not_loaded:
                if not config.get('cpu_allow_missing_llm', False):
                    raise RuntimeError(
                        'CPU LLM checkpoint is missing non-adapter weights: '
                        f'{not_loaded[:12]}. Check the checkpoint format or set '
                        'cpu_allow_missing_llm=True to explicitly accept this.'
                    )
                print('### Warning: missing LLM keys:', not_loaded[:12], flush=True)
            if report['unexpected']:
                print('### Warning: unused LLM checkpoint keys:',
                      report['unexpected'][:12], flush=True)
            if hasattr(model, 'tie_weights'):
                # Match the common post-load weight-tying step used by HF.
                model.tie_weights()
        else:
            if sequential and source is None:
                print('### No supported local LLM shard/index found; '
                      'using compatible from_pretrained loader', flush=True)
            kwargs = {'config': text_config}
            if use_mmap:
                from models.utils import load_sharded_bin_mmap
                kwargs['state_dict'] = load_sharded_bin_mmap(rpath)

            if config.get('llm_low_cpu_mem_usage', False):
                # Requires a Transformers version that supports the flag, plus
                # Accelerate. Keep off by default for legacy custom models.
                kwargs['low_cpu_mem_usage'] = True

            dtype_name = (config.get('cpu_llm_dtype', config.get('llm_dtype'))
                          if llm_device.type == 'cpu'
                          else config.get('gpu_llm_dtype', config.get('llm_dtype')))
            if dtype_name is not None:
                if dtype_name == 'auto':
                    kwargs['torch_dtype'] = 'auto'
                elif dtype_name in ('float16', 'bfloat16', 'float32'):
                    kwargs['torch_dtype'] = getattr(torch, dtype_name)
                else:
                    raise ValueError(f"Unsupported llm_dtype: {dtype_name!r}")

            model = LlamaForCausalLM.from_pretrained(rpath, **kwargs)
            kwargs.pop('state_dict', None)

        # Base Vicuna has no output adapters; they are initialized here and
        # subsequently restored from finetune_lynx.pt by load_pretrained().
        # A tokenizer helper's reported new-token count is NOT authoritative:
        # it may report 1 without changing the actual vocabulary length.
        input_embeddings = model.get_input_embeddings().weight
        actual_vocab = int(input_embeddings.shape[0])
        tokenizer_vocab = len(self.tokenizer)
        actually_added = tokenizer_vocab - actual_vocab
        if num_new_tokens != max(actually_added, 0):
            print(f'### Lynx tokenizer: helper reported {num_new_tokens} new token(s), '
                  f'but tokenizer size={tokenizer_vocab} and model vocab={actual_vocab}; '
                  'using actual vocabulary size.', flush=True)
        if actually_added < 0:
            raise ValueError(
                f'Tokenizer has {tokenizer_vocab} entries but LLM embeddings '
                f'have {actual_vocab}. Refusing to shrink pretrained embeddings.'
            )
        if actually_added > 0:
            print(f'### Lynx: extending LLM embeddings from {actual_vocab} to '
                  f'{tokenizer_vocab} (new tokens: {actually_added})', flush=True)
            # Preserve old embeddings exactly, initializing *only* the newly
            # introduced rows to the mean, not the last existing token row.
            model.resize_token_embeddings(tokenizer_vocab)
            with torch.no_grad():
                in_weight = model.get_input_embeddings().weight
                out_weight = model.get_output_embeddings().weight
                in_weight[actual_vocab:tokenizer_vocab].copy_(
                    in_weight[:actual_vocab].float().mean(dim=0, keepdim=True)
                    .to(dtype=in_weight.dtype)
                )
                if out_weight.shape[0] >= tokenizer_vocab:
                    out_weight[actual_vocab:tokenizer_vocab].copy_(
                        out_weight[:actual_vocab].float().mean(dim=0, keepdim=True)
                        .to(dtype=out_weight.dtype)
                    )

        # Both tokenizer and model must use IDs inside the FINAL vocabulary.
        final_vocab = int(model.get_input_embeddings().weight.shape[0])
        pad_id = _ensure_valid_tokenizer_pad(self.tokenizer)
        if not _valid_vocab_id(pad_id, final_vocab):
            raise ValueError(f'Invalid final pad_token_id={pad_id} for '
                             f'model vocabulary size {final_vocab}')
        model.model.padding_idx = pad_id
        model.config.pad_token_id = pad_id
        if hasattr(model.model, 'config'):
            model.model.config.pad_token_id = pad_id
        # Some legacy custom model constructors do not create this attribute.
        if getattr(model, 'generation_config', None) is None:
            from transformers import GenerationConfig
            model.generation_config = GenerationConfig.from_model_config(model.config)
        model.generation_config.pad_token_id = pad_id
        for token_name in ('eos_token_id', 'bos_token_id'):
            candidate = getattr(self.tokenizer, token_name, None)
            if _valid_vocab_id(candidate, final_vocab):
                setattr(model.config, token_name, candidate)
                setattr(model.generation_config, token_name, candidate)
        print(f'### Lynx tokenizer IDs: vocab={final_vocab}, '
              f'pad={pad_id}, eos={getattr(self.tokenizer, "eos_token_id", None)}',
              flush=True)
        missing_keys = [n for n, _ in model.named_parameters() if 'adapter' in n]
        self.text_width = model.config.hidden_size

        if freeze_params:
            print('### Freeze LLM', flush=True)
            for name, param in model.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad_(False)

        # Original 2023 LLaMA attention's handwritten score/softmax path can
        # yield a non-finite token row even with all-finite checkpoint weights.
        # Install a compatible SDPA implementation before training/generation;
        # this leaves adapter and other parameter names unchanged.
        self._lynx_stable_attention_layers = _lynx_enable_stable_legacy_attention(
            model, backend=config.get('legacy_llama_attention_backend', 'sdpa'),
            check_qkv=config.get('legacy_llama_attention_check_qkv', True),
            repair_rope=config.get('legacy_llama_repair_rope_buffers', True))
        self._lynx_stable_rmsnorm_layers = _lynx_enable_stable_rmsnorm(
            model, enabled=config.get('legacy_llama_stable_rmsnorm', True))

        return model, missing_keys

    def build_bridge(self, config, load_params=False):
        """
        Bridge for Vision to Text
        """
        print("### Building Bridge", flush=True)
        missing_keys = []

        if config['bridge'] == 'resampler':
            from models.bridges.resampler import PerceiverResampler
            model = PerceiverResampler(self.vision_width, self.text_width,
                                       depth=config["bridge_depth"], num_latents=config["num_bridge_tokens"])
            assert load_params is False, "no param to load for Resampler"
            missing_keys = [n for (n, p) in model.named_parameters()]
        else:
            raise NotImplementedError("Bridge: ", config['bridge'])

        if load_params:
            print("missing_keys: ", missing_keys, flush=True)

        return model, missing_keys

    def update_init_params(self, missing_keys=None):
        """Track missing trainable weights without quadratic membership scans."""
        if missing_keys is not None:
            if not isinstance(missing_keys, list):
                raise TypeError('missing_keys must be a list')
            self.init_params.extend(missing_keys)

        present = {name for name, _ in self.named_parameters()}
        self.init_params = list(dict.fromkeys(
            name for name in self.init_params if name in present
        ))

    @staticmethod
    def _read_checkpoint(path, mmap=True, weights_only=True):
        """Read a PyTorch checkpoint using CPU memory mapping when available.

        For ZIP-format torch.save files, mmap delays loading tensor bytes
        until they are used; it does not magically index arbitrary pickle
        files. Legacy-format checkpoints fall back to conventional torch.load.
        """
        supported = inspect.signature(torch.load).parameters
        kwargs = {'map_location': 'cpu'}
        if 'weights_only' in supported:
            kwargs['weights_only'] = weights_only
        elif weights_only:
            print('### Warning: torch.load lacks weights_only', flush=True)
        if mmap and 'mmap' in supported:
            try:
                return torch.load(path, mmap=True, **kwargs)
            except RuntimeError as exc:
                # Only fall back for explicitly unsupported storage formats.
                error_text = str(exc).lower()
                if not any(word in error_text for word in
                           ('mmap', 'zip', 'legacy', 'old format')):
                    raise
                print('### Checkpoint cannot be memory-mapped; standard CPU load', flush=True)
        return torch.load(path, **kwargs)

    @staticmethod
    def _resolve_cpu_weights(path):
        """Locate a checkpoint or a Hugging Face-style sharded weight index.

        Returns (list_of_paths, mapping_from_tensor_name_to_shard_or_None).
        Shards are read in deterministic order, serially, never concurrently.
        """
        path = Path(path)
        if path.is_dir():
            index_names = ('model.safetensors.index.json',
                           'pytorch_model.bin.index.json')
            index_path = next((path / name for name in index_names
                               if (path / name).is_file()), None)
            if index_path is None:
                for filename in ('model.safetensors', 'pytorch_model.bin'):
                    if (path / filename).is_file():
                        return ([path / filename], None)
                return None
        elif path.name.endswith('.index.json'):
            index_path = path
        elif path.is_file():
            return ([path], None)
        else:
            return None

        with index_path.open('r', encoding='utf-8') as f:
            index = json.load(f)
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f'Invalid weight_map in checkpoint index: {index_path}')

        root = index_path.parent.resolve()
        files = []
        for shard_name in dict.fromkeys(weight_map.values()):
            if not isinstance(shard_name, str):
                raise ValueError('A checkpoint shard name must be a string')
            shard_path = (root / shard_name).resolve()
            if os.path.commonpath((str(root), str(shard_path))) != str(root):
                raise ValueError(f'Shard path escapes checkpoint folder: {shard_name}')
            if not shard_path.is_file():
                raise FileNotFoundError(f'Checkpoint shard missing: {shard_path}')
            files.append(shard_path)
        # Preserve the manifest order for deterministic serial loading.
        return (files, {key: (root / shard).resolve()
                        for key, shard in weight_map.items()})

    @staticmethod
    def _canonical_weight_key(key, *, normalize_prefix, expected_keys):
        # `str.lstrip('model.')` in the original source was a correctness bug.
        # Strip only an actual outer model wrapper, and only if the unwrapped
        # name refers to a real parameter/buffer of this architecture.
        if normalize_prefix and key.startswith('model.'):
            candidate = key[len('model.'):]
            if candidate in expected_keys:
                return candidate
        return key

    @staticmethod
    def _copy_cpu_parameter(dst, src, *, name, rebind):
        if not isinstance(src, torch.Tensor):
            raise TypeError(f'{name} is not a tensor: {type(src)}')
        if src.device.type != 'cpu':
            raise ValueError('Sequential weight reader expects CPU checkpoint tensors')
        if dst.device.type not in ('cpu', 'cuda'):
            raise ValueError(f'Unsupported destination device for {name}: {dst.device}')
        if rebind and dst.device.type != 'cpu':
            raise ValueError('Zero-copy storage rebind needs a CPU destination; '
                             'GPU destinations require a direct parameter copy')
        if dst.shape != src.shape:
            raise ValueError(f'Shape mismatch for {name}: model {tuple(dst.shape)}, '
                             f'checkpoint {tuple(src.shape)}')
        with torch.no_grad():
            if rebind:
                # Zero-copy replacement: skips the full CPU tensor copy.
                # Intended only before an optimizer is constructed.
                dst.data = src
            else:
                # copy_ supports CPU->CUDA directly; no full GPU checkpoint
                # materialization is needed when reloading an existing model.
                dst.copy_(src)

    def _load_cpu_weights(self, module, source, *, normalize_prefix,
                          mmap=True, weights_only=True, rebind=False,
                          interpolate_vision=False):
        """Sequential CPU loader: each shard/tensor is processed once.

        Unlike loading an aggregated state_dict, this holds only the active
        PyTorch shard's metadata (or one SafeTensors tensor) at a time.
        Uses state_dict(keep_vars=True) to reference *live* parameters and
        persistent buffers without duplicating their memory. No worker pool.

        rebind=True uses file-backed storages when loading a new model BEFORE
        optimizer creation. Never use it when reloading an existing optimizer:
        replacement storages could break parameter aliases or optimizer state.
        Non-matching shared storages are rejected by the guard below.
        """
        shard_paths, weight_map = source
        if rebind and getattr(module, '_is_replica', False):
            raise ValueError('Cannot rebind parameters of an existing model replica')

        destinations = module.state_dict(keep_vars=True)
        if any(not isinstance(t, torch.Tensor) or t.device.type not in ('cpu', 'cuda')
               for t in destinations.values()):
            raise ValueError('Sequential loader supports only CPU/CUDA tensor state; '
                             'use the legacy loader for custom extra_state')
        if rebind and any(t.device.type != 'cpu' for t in destinations.values()):
            raise ValueError('Storage rebinding is supported only before CPU->GPU placement')

        expected = set(destinations)
        if rebind:
            # Distinct parameters can intentionally share the same underlying
            # storage. Independent .data assignments would destroy that alias.
            # Multiple state_dict names for the SAME Parameter remain safe.
            aliases = {}
            for name, tensor in destinations.items():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError('CPU zero-copy mode supports tensor state only')
                if tensor.numel() == 0:
                    continue
                location = tensor.untyped_storage().data_ptr()
                if location in aliases and aliases[location][0] is not tensor:
                    raise ValueError(
                        f'Zero-copy rebind would break shared storage between '
                        f'{aliases[location][1]} and {name}; disable cpu_rebind_weights'
                    )
                aliases[location] = (tensor, name)
        seen = set()
        unexpected = set()
        seen_manifest = set()

        def copy_weight(key, value, shard_path):
            name = self._canonical_weight_key(key, normalize_prefix=normalize_prefix,
                                              expected_keys=expected)
            if weight_map is not None:
                manifest_name = weight_map.get(key)
                if manifest_name != shard_path:
                    # Indexed checkpoints must not quietly load tensors from a
                    # different file or accept unindexed extra parameters.
                    raise ValueError(f'Checkpoint index mismatch for tensor {key!r}')
                seen_manifest.add(key)
            if name not in destinations:
                unexpected.add(name)
                return
            if name in seen:
                raise ValueError(f'Duplicate parameter in checkpoint shards: {name}')
            if interpolate_vision and name == 'vision_encoder.pos_embed':
                one_key = {name: value}
                interpolate_pos_embed(module.vision_encoder, one_key,
                                      pos_embed_name=name)
                value = one_key[name]
            self._copy_cpu_parameter(destinations[name], value, name=name, rebind=rebind)
            seen.add(name)

        for shard_path in shard_paths:
            print(f'### Loading CPU shard: {shard_path.name}', flush=True)
            if shard_path.suffix == '.safetensors':
                try:
                    from safetensors import safe_open
                except ImportError as exc:
                    raise ImportError('Install safetensors to read .safetensors shards') from exc
                # get_tensor returns one tensor at a time; no combined weights.
                with safe_open(str(shard_path), framework='pt', device='cpu') as shard:
                    for key in shard.keys():
                        value = shard.get_tensor(key)
                        copy_weight(key, value, shard_path)
                        del value
            else:
                checkpoint = self._read_checkpoint(str(shard_path),
                                                   mmap=mmap, weights_only=weights_only)
                state_dict = checkpoint.get('model', checkpoint)
                if not isinstance(state_dict, dict):
                    raise TypeError(f'Expected weight dictionary in {shard_path}')
                for key, value in state_dict.items():
                    copy_weight(key, value, shard_path)
                del checkpoint, state_dict

        if weight_map is not None:
            absent = set(weight_map) - seen_manifest
            if absent:
                raise ValueError(f'Indexed checkpoint omits weights: {sorted(absent)[:10]}')
        return {'missing': sorted(expected - seen),
                'unexpected': sorted(unexpected),
                'loaded': len(seen), 'shards': len(shard_paths),
                'adapter_expected': sum('adapter' in key.lower() for key in expected),
                'adapter_loaded': sum('adapter' in key.lower() for key in seen),
                'adapter_loaded_keys': sorted(k for k in seen if 'adapter' in k.lower()),
                'adapter_missing': sorted(key for key in expected - seen
                                          if 'adapter' in key.lower())}

    def load_pretrained(self, ckpt_rpath, config):
        """Load the full Lynx checkpoint on CPU.

        CPU/GPU loading: stream CPU memory-mapped shards before GPU placement.
        For ordinary PyTorch files, mmap maps lazy tensor storage and each
        parameter is copied once. For SafeTensors, fetch one tensor at a time.

        Config keys:
          cpu_sequential_checkpoint: True (default)
          checkpoint_mmap: True (default)
          checkpoint_weights_only: True (default)
          cpu_rebind_weights: False (true opts into zero-copy storage rebinding)
          checkpoint_assign: False (legacy single-file torch.load path only)
        """
        print('### load params from: ', ckpt_rpath, flush=True)
        vision_device = getattr(self, 'vision_device', torch.device('cpu'))
        llm_device = getattr(self, 'llm_device', torch.device('cpu'))
        gpu_run = vision_device.type == 'cuda' or llm_device.type == 'cuda'
        sequential = (config.get('gpu_sequential_checkpoint', True) if gpu_run
                      else config.get('cpu_sequential_checkpoint', True))
        rebind = config.get('checkpoint_zero_copy_load',
                            config.get('gpu_rebind_weights', False) if gpu_run
                            else config.get('cpu_rebind_weights', False))
        print(f'### {"GPU" if gpu_run else "CPU"} checkpoint mode: '
              f'{"sequential mmap staging" if sequential else "legacy loader"}',
              flush=True)
        source = self._resolve_cpu_weights(ckpt_rpath)
        if source is None:
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_rpath}')
        if sequential:
            if config.get('checkpoint_assign', False):
                print('### checkpoint_assign ignored in sequential mode; '
                      'use cpu_rebind_weights / gpu_rebind_weights for zero-copy loading', flush=True)
            report = self._load_cpu_weights(
                self, source, normalize_prefix=True,
                mmap=config.get('checkpoint_mmap', True),
                weights_only=config.get('checkpoint_weights_only', True),
                rebind=rebind,
                interpolate_vision=True,
            )
            print(f"### Loaded {report['loaded']} tensors from "
                  f"{report['shards']} CPU file(s)", flush=True)
            self._lynx_checkpoint_load_report = report
            print(f"### Checkpoint adapter coverage: "
                  f"{report['adapter_loaded']}/{report['adapter_expected']} "
                  "expected adapter tensors loaded", flush=True)
            if report['unexpected']:
                adapter_only = [k for k in report['unexpected'] if 'adapter' in k.lower()]
                other = [k for k in report['unexpected'] if k not in adapter_only]
                if adapter_only:
                    print(f'### WARNING: {len(adapter_only)} checkpoint adapter tensors '
                          'did not match any model parameter and WERE NOT LOADED. '
                          f"use_adapter={config.get('use_adapter', False)!r}, "
                          f"adapter_freq={config.get('adapter_freq', -1)!r}. "
                          'Check model configuration and checkpoint architecture. '
                          'Set use_adapter: true and the training adapter_freq '
                          'when loading an adapter-trained checkpoint.', flush=True)
                    print('### First unmatched adapter keys:', adapter_only[:5], flush=True)
                    checkpoint_layers = _adapter_layer_indices(
                        list(report['unexpected']) + report['adapter_loaded_keys']
                    )
                    model_layers = _adapter_layer_indices(
                        n for n, _ in self.named_parameters())
                    hint = _adapter_stride_hint(checkpoint_layers)
                    print(f'### Adapter layers: checkpoint={checkpoint_layers}; '
                          f'constructed model={model_layers}; '
                          f'suggested adapter_freq={hint}', flush=True)
                    if config.get('strict_checkpoint_adapters', False):
                        raise RuntimeError(
                            'Checkpoint contains unmatched adapter weights. '
                            f'Checkpoint layers={checkpoint_layers}, '
                            f'model layers={model_layers}, '
                            f'configured adapter_freq={config.get("adapter_freq")}. '
                            f'Suggested adapter_freq={hint}; verify training architecture '
                            'before changing CROSSVAL.yaml. Do not disable strict checking '
                            'if all fine-tuned adapters are needed.'
                        )
                if other:
                    print(f'### Unexpected non-adapter checkpoint keys ({len(other)}):',
                          other[:12], flush=True)
            if config.get('strict_checkpoint_adapters', False):
                missing_adapter = report['adapter_missing']
                if missing_adapter:
                    raise RuntimeError(
                        f"Checkpoint is missing {len(missing_adapter)} adapter "
                        f"parameters expected by use_adapter={config.get('use_adapter')}, "
                        f"adapter_freq={config.get('adapter_freq')}. "
                        f"Examples: {missing_adapter[:5]}. "
                        "Check the original training adapter architecture."
                    )
                if config.get('use_adapter', False) and report['adapter_loaded'] == 0:
                    raise RuntimeError(
                        'use_adapter=True but no adapter tensor was restored. '
                        'Verify the checkpoint and adapter_freq.'
                    )
            missing_keys = report['missing']
        else:
            # The original state_dict path remains for custom modules that
            # implement their own PyTorch load_state_dict hooks. It requires
            # a single PyTorch checkpoint, not an indexed shard directory.
            files, weight_map = source
            if len(files) != 1 or weight_map is not None or files[0].suffix == '.safetensors':
                raise ValueError('Legacy mode requires one .pt/.pth/.bin checkpoint')
            checkpoint = self._read_checkpoint(
                str(files[0]), mmap=config.get('checkpoint_mmap', True),
                weights_only=config.get('checkpoint_weights_only', True),
            )
            state_dict = checkpoint.get('model', checkpoint)
            if not isinstance(state_dict, dict):
                raise TypeError('Expected state_dict or checkpoint["model"]')
            expected = set(self.state_dict())
            state_dict = {
                self._canonical_weight_key(k, normalize_prefix=True,
                                           expected_keys=expected): v
                for k, v in state_dict.items()
            }
            interpolate_pos_embed(self.vision_encoder, state_dict,
                                  pos_embed_name='vision_encoder.pos_embed')
            load_kwargs = {'strict': False}
            if config.get('checkpoint_assign', False):
                if 'assign' not in inspect.signature(self.load_state_dict).parameters:
                    raise RuntimeError('checkpoint_assign requires PyTorch with assign support')
                if any(p.device.type != 'cpu' for p in self.parameters()):
                    raise ValueError('checkpoint_assign requires model on CPU')
                load_kwargs['assign'] = True
            msg = self.load_state_dict(state_dict, **load_kwargs)
            if config.get('strict_checkpoint_adapters', False):
                unexpected = [k for k in msg.unexpected_keys if 'adapter' in k.lower()]
                missing_adapter = [k for k in msg.missing_keys if 'adapter' in k.lower()]
                if unexpected or missing_adapter:
                    raise RuntimeError(
                        'Legacy checkpoint adapter mismatch: '
                        f'unexpected={unexpected[:5]}, missing={missing_adapter[:5]}. '
                        'Restore the training-time use_adapter/adapter_freq values.'
                    )
            missing_keys = msg.missing_keys
            del checkpoint, state_dict

        if config.get('freeze_vit', self.freeze_vit):
            missing_keys = [k for k in missing_keys if not k.startswith('vision_encoder.')]
        if config.get('freeze_llm', self.freeze_llm):
            missing_keys = [
                k for k in missing_keys
                if not k.startswith('LLM.') or
                (config.get('use_adapter', False) and 'adapter' in k)
            ]
        # Diagnose corrupted fine-tuned weights before CUDA placement and
        # before the first forward pass. These small components carry the
        # checkpoint-specific adaptation; scanning the frozen 7B base model
        # would duplicate work and substantially slow initialization.
        if config.get('check_checkpoint_finite', True):
            checked = 0
            for name, tensor in self.named_parameters():
                if 'adapter' not in name.lower() and not name.startswith('bridge.'):
                    continue
                checked += 1
                with torch.no_grad():
                    if not bool(torch.isfinite(tensor).all()):
                        bad = int((~torch.isfinite(tensor)).sum().item())
                        raise RuntimeError(
                            f'Checkpoint/model parameter {name!r} contains {bad} '
                            'non-finite values BEFORE training. Repair the source '
                            'weights/checkpoint; changing the loss cannot fix this.'
                        )
            print(f'### Checkpoint finite-weight preflight: {checked} '
                  'adapter/bridge tensors verified', flush=True)

        # Optional V13 full base-model preflight. Adapter/bridge checks alone
        # cannot reveal invalid pretrained LLaMA weights. This scans one small
        # CPU tensor slice at a time; no entire-model GPU copy is created.
        # Repair only corrupted procedural RoPE buffers BEFORE the finite
        # parameter+buffer scan. No learned or checkpoint-trained weights changed.
        self._lynx_rotary_report = _lynx_validate_llama_rotary_buffers(
            self.LLM, repair=config.get('legacy_llama_repair_rope_buffers', True))
        if config.get('crossval_check_base_llm_finite', False):
            self.verify_llm_base_parameters_finite()

        missing = set(missing_keys)
        self.init_params = [name for name in self.init_params if name in missing]

    def verify_llm_base_parameters_finite(self, chunk_elements=1_048_576):
        """Check ALL pretrained LLaMA tensors without allocating a second copy.

        Adapters and the bridge are checked by the existing checkpoint preflight.
        This additional test covers base attention, MLP, norms, and lm_head.
        Cost: one CPU memory scan of the base LLM during model construction.
        """
        checked = 0
        elements = 0
        with torch.no_grad():
            for name, parameter in self.LLM.named_parameters():
                if 'adapter' in name.lower():
                    continue
                checked += 1
                flattened = parameter.detach().reshape(-1)
                for start in range(0, flattened.numel(), chunk_elements):
                    piece = flattened[start:start + chunk_elements]
                    if (piece.is_floating_point() or piece.is_complex()) and not bool(torch.isfinite(piece).all()):
                        bad = int((~torch.isfinite(piece)).sum().item())
                        raise RuntimeError(
                            'Non-finite PRETRAINED LLM parameter detected BEFORE training: '
                            f'LLM.{name}, slice={start}:{start+piece.numel()}, '
                            f'bad_values={bad}. Re-download/verify the base Vicuna '
                            'checkpoint; do not replace invalid weights with zeros.'
                        )
                elements += flattened.numel()
        checked_buffers = 0
        with torch.no_grad():
            for name, buffer in self.LLM.named_buffers():
                if buffer is None or not buffer.is_floating_point():
                    continue
                checked_buffers += 1
                flattened = buffer.detach().reshape(-1)
                for start in range(0, flattened.numel(), chunk_elements):
                    piece = flattened[start:start + chunk_elements]
                    if not bool(torch.isfinite(piece).all()):
                        raise RuntimeError(
                            f'Non-finite LLaMA BUFFER: LLM.{name}, '
                            'the former parameter-only preflight missed buffers; '
                            'verify rotary tables and the source checkpoint.')
        print(f'### Lynx base LLM preflight: {checked} parameter tensors, '
              f'{elements:,} parameter elements finite; '
              f'{checked_buffers} floating-point buffers also finite', flush=True)
        return checked

    def _encode_video_frames(self, frames):
        assert frames.dim() == 5  # (bsz, frame_len, c, h, w)
        bsz = frames.shape[0]

        # Encode Each Frame
        frames = rearrange(frames, 'b f c h w -> (b f) c h w')
        frame_embeds = self.vision_encoder(frames)
        frame_embeds = rearrange(frame_embeds, '(b f) p d -> b f p d', b=bsz)

        if self.add_frame_pos:
            frame_embeds = frame_embeds + self.absolute_frame_pos_embed

        return frame_embeds

    def get_video_embeds(self, frames):
        assert frames.dim() == 5  # (bsz, frame_len, c, h, w)

        if self.video_encoding == 'concate':
            video_embeds = self._encode_video_frames(frames)
            video_embeds = rearrange(video_embeds, 'b f p d -> b (f p) d')
            video_atts = torch.ones(video_embeds.shape[:-1], device=video_embeds.device, dtype=torch.long)
        else:
            raise NotImplementedError

        return video_embeds, video_atts

    def get_image_embeds(self, image):
        assert image.dim() == 4
        image_embeds = self.vision_encoder(image)
        image_atts = torch.ones(image_embeds.shape[:-1], device=image_embeds.device, dtype=torch.long)
        return image_embeds, image_atts  # full attention

    @staticmethod
    def _module_device(module, fallback=torch.device('cpu')):
        """Find device even for parameterless modules."""
        first = next(module.parameters(), None)
        if first is not None:
            return first.device
        first = next(module.buffers(), None)
        return first.device if first is not None else fallback

    def get_vision_embeds(self, vision):
        target = self._module_device(
            self.vision_encoder, getattr(self, 'vision_device', vision.device)
        )
        # FP16/BF16 vision weights require images in the same dtype.
        # Converting on the destination GPU avoids an extra full image copy.
        vision_param = next(self.vision_encoder.parameters(), None)
        dtype = (vision_param.dtype if vision_param is not None and
                 vision.is_floating_point() else vision.dtype)
        if vision.device != target or vision.dtype != dtype:
            vision = vision.to(device=target, dtype=dtype, non_blocking=True)
        if vision.dim() == 5:  # encode video
            return self.get_video_embeds(vision)
        assert vision.dim() == 4
        return self.get_image_embeds(vision)

    def embed_tokens(self, input_ids):
        if self.use_dec_only:
            text_embeds = self.LLM.model.embed_tokens(input_ids)
        else:
            raise NotImplementedError

        return text_embeds

    def get_mixed_v2t_feats(self, image, video):

        assert ((image is None) and (video is None)) is False

        with torch.set_grad_enabled(not self.freeze_vit):
            if image is not None:
                image_embeds, image_atts = self.get_vision_embeds(image)

            if video is not None:
                video_embeds, video_atts = self.get_vision_embeds(video)

        bridge_device = self._module_device(
            self.bridge, getattr(self, 'vision_device', torch.device('cpu'))
        )
        img_v2t_feats, img_v2t_atts = None, None
        if image is not None:
            img_v2t_feats, img_v2t_atts = self.bridge(
                vision_embeds=image_embeds.to(bridge_device, dtype=next(self.bridge.parameters()).dtype, non_blocking=True),
                vision_atts=image_atts.to(bridge_device, non_blocking=True),
            )
            img_v2t_atts = _lynx_resampler_attention_mask(
                img_v2t_feats, img_v2t_atts, name='mixed_image_bridge_mask')

        video_v2t_feats, video_v2t_atts = None, None
        if video is not None:
            video_v2t_feats, video_v2t_atts = self.bridge(
                vision_embeds=video_embeds.to(bridge_device, dtype=next(self.bridge.parameters()).dtype, non_blocking=True),
                vision_atts=video_atts.to(bridge_device, non_blocking=True),
            )
            video_v2t_atts = _lynx_resampler_attention_mask(
                video_v2t_feats, video_v2t_atts, name='mixed_video_bridge_mask')

        def _concate(video, image):
            assert ((video is None) and (image is None)) is False
            if video is None:
                return image
            elif image is None:
                return video
            else:
                return torch.concat([video, image], dim=0)  # video first

        return _concate(video_v2t_feats, img_v2t_feats), _concate(video_v2t_atts, img_v2t_atts)

    @torch.inference_mode()
    def generate(self, vision_input, input_ids, input_atts,
                 use_nucleus_sampling=False, num_beams=5, max_new_tokens=64, min_length=2, top_p=0.9, top_k=3,
                 repetition_penalty=1.0, no_repeat_ngram_size=3,
                 length_penalty=1.0, num_return_sequences=1, temperature=1, apply_lemmatizer=False,
                 invalid_logits_policy='repair'):
        
        # Keep tensors on the embedding model's device. The original code
        # needlessly transferred embeddings CUDA -> CPU before generate().
        embed_device = self.LLM.model.embed_tokens.weight.device
        input_ids = input_ids.to(embed_device, non_blocking=True)
        text_embeds = self.embed_tokens(input_ids)
        target_device = text_embeds.device
        input_atts = _lynx_normalize_attention_mask(
            input_atts.to(target_device, non_blocking=True),
            name='input_token_attention_mask')

        if vision_input is not None:
            # Works when the vision encoder / bridge are on different devices.
            vision_device = self._module_device(
                self.vision_encoder, getattr(self, 'vision_device', vision_input.device)
            )
            vision_embeds, vision_atts = self.get_vision_embeds(vision_input)
            bridge_device = self._module_device(
                self.bridge, getattr(self, 'vision_device', vision_device)
            )
            bridge_param = next(self.bridge.parameters(), None)
            bridge_dtype = bridge_param.dtype if bridge_param is not None else vision_embeds.dtype
            vision_embeds = vision_embeds.to(
                device=bridge_device, dtype=bridge_dtype, non_blocking=True)
            vision_atts = vision_atts.to(bridge_device, non_blocking=True)
            v2t_feats, v2t_atts = self.bridge(
                vision_embeds=vision_embeds, vision_atts=vision_atts
            )

            v2t_feats = v2t_feats.to(
                device=target_device, dtype=text_embeds.dtype, non_blocking=True
            )
            # Fixed Perceiver latent queries are all valid. The legacy bridge's
            # second output may contain non-binary garbage; never use its
            # numeric values as attention indicators/rotary positions.
            v2t_atts = _lynx_resampler_attention_mask(
                v2t_feats, v2t_atts, name='bridge_attention_mask')

            inputs_embeds = torch.cat((v2t_feats, text_embeds), dim=1)
            attention_mask = torch.cat((v2t_atts, input_atts), dim=1)
        else:
            inputs_embeds = text_embeds
            attention_mask = input_atts

        if not bool(torch.isfinite(inputs_embeds).all()):
            # NaN activations (often from an FP16 vision encoder or training)
            # cannot be corrected by filtering the final vocabulary logits.
            nonfinite = int((~torch.isfinite(inputs_embeds)).sum())
            raise RuntimeError(
                f'Lynx numerical error: {nonfinite} non-finite multimodal input '
                f'embedding values before decoding (dtype={inputs_embeds.dtype}, '
                f'device={inputs_embeds.device}). Check bridge/vision activations '
                'and consider gpu_vision_dtype: float32.'
            )
        if attention_mask.ndim != 2 or attention_mask.shape[:2] != inputs_embeds.shape[:2]:
            raise ValueError('Lynx multimodal generation shape mismatch: '
                             f'inputs_embeds={tuple(inputs_embeds.shape)}, '
                             f'attention_mask={tuple(attention_mask.shape)}')

        # Identify the *active* backend in Colab logs (printed once per model).
        if not getattr(self, '_lynx_direct_decoder_announced', False):
            print(f'### Lynx decoding backend: {LYNX_GENERATION_PATCH_VERSION}', flush=True)
            self._lynx_direct_decoder_announced = True

        # Call Lynx's direct legacy decoder. Hugging Face GenerationMixin in
        # current Colab is incompatible with this project's 2023 tuple KV cache.
        # Bypassing it prevents stale position_ids from expanding [1,1] to
        # [47,1] inside rotary attention during cached decoding.
        outputs = _lynx_generate_legacy_tokens(
            self.LLM,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            invalid_logits_policy=invalid_logits_policy,
        )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def prepare_crossval_training(self):
        """Make the legacy LLaMA training path deterministic and memory-safe.

        `cross_val.py` calls the LLM with use_cache=False explicitly, so this
        only changes the default. The direct inference decoder still explicitly
        uses tuple KV caching and remains unaffected.
        """
        for component in (self.LLM, getattr(self.LLM, "model", None)):
            if component is not None and getattr(component, "config", None) is not None:
                component.config.use_cache = False
        print("### Lynx: training KV cache disabled; inference cache preserved", flush=True)
        if getattr(self, '_lynx_crossval_activation_checkpointing', True):
            n = _lynx_enable_decoder_activation_checkpointing(self.LLM)
            print('### Lynx RAM: non-reentrant decoder activation checkpointing '
                  f'installed on {n} new layer(s); preserves input gradients', flush=True)
        return self

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        if not hasattr(self, "lemmatizer"):
            import spacy
            self.lemmatizer = spacy.load("en_core_web_sm")

        return [apply(answer) for answer in answers]
