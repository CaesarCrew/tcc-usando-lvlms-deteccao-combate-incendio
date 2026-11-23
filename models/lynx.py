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

import inspect
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from einops import rearrange

from timm.models.layers import trunc_normal_

from additional import build_tokenizer

from .vits.eva_vit import interpolate_pos_embed


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

        self.freeze_vit = freeze_vit
        self.freeze_llm = freeze_llm
        self.init_params = []

        # Construct and load weights on the CPU to avoid temporary CUDA copies
        # and to support sequential memory-mapped checkpoint loading.
        self.vision_encoder, missing_keys = self.build_vision_encoder(config, freeze_params=freeze_vit)
        self.update_init_params([f'vision_encoder.{k}' for k in missing_keys])

        self.tokenizer, num_new_tokens = build_tokenizer(config['LLM'], use_left_pad=config['use_left_pad'])
        llm_on_cpu = self.llm_device.type == 'cpu'
        sequential_llm = (config.get('cpu_sequential_llm', True) if llm_on_cpu
                          else config.get('gpu_sequential_llm', False))
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

    def place_model_on_runtime_devices(self, config=None):
        """Place ViT+bridge together; move LLM separately, once after loading.

        Call this again if the external driver moves the entire model with
        model.to(...), which otherwise overrides mixed-device placement.
        Use before constructing an optimizer.
        """
        config = config or {}
        vision_device = self.vision_device
        llm_device = self.llm_device
        if vision_device == llm_device:
            # One pass when both submodels use the same device.
            self.to(vision_device)
        else:
            self.vision_encoder.to(vision_device)
            self.bridge.to(vision_device)
            if hasattr(self, 'absolute_frame_pos_embed'):
                parameter = self.absolute_frame_pos_embed
                self.absolute_frame_pos_embed = nn.Parameter(
                    parameter.detach().to(vision_device),
                    requires_grad=parameter.requires_grad,
                )
            self.LLM.to(llm_device)

        # Opt in to FP16 or BF16 GPU weights. Defaults remain float32/native
        # to preserve numerical behavior for older custom LLaMA versions.
        if llm_device.type == 'cuda' and config.get('gpu_llm_dtype'):
            name = config['gpu_llm_dtype']
            if name in ('float16', 'bfloat16', 'float32'):
                dtype = getattr(torch, name)
                if name == 'bfloat16' and not torch.cuda.is_bf16_supported():
                    raise ValueError('gpu_llm_dtype=bfloat16 requires a BF16-capable GPU')
                self.LLM.to(dtype=dtype)
            elif name != 'auto':
                raise ValueError(f'Unsupported gpu_llm_dtype: {name!r}')
        print('### Components placed: '
              f'vision/bridge={vision_device}, LLM={llm_device}', flush=True)
        return self

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

        rpath = os.path.join('data', config['LLM'])
        if not os.path.isdir(rpath):
            raise FileNotFoundError(f"LLM directory not found: {rpath}")

        text_config = LlamaConfig.from_json_file(os.path.join(rpath, 'config.json'))
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
                          else config.get('gpu_sequential_llm', False))
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
            report = self._load_cpu_weights(
                model, source, normalize_prefix=False,
                mmap=config.get('checkpoint_mmap', True),
                weights_only=config.get('checkpoint_weights_only', True),
                rebind=(config.get('cpu_rebind_weights', False)
                        if llm_device.type == 'cpu'
                        else config.get('gpu_rebind_weights', False)),
                interpolate_vision=False,
            )
            # LLaMA adapters are intentionally absent from vanilla Vicuna
            # checkpoints. Other missing keys may be from custom model code.
            not_loaded = [k for k in report['missing'] if 'adapter' not in k]
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

        model.model.padding_idx = self.tokenizer.pad_token_id
        missing_keys = [n for n, _ in model.named_parameters() if 'adapter' in n]
        self.text_width = model.config.hidden_size

        if num_new_tokens > 0:
            print('### LLM Vocab Size: ', model.config.vocab_size, flush=True)
            print('### num_new_tokens: ', num_new_tokens, flush=True)
            vocab_size = model.config.vocab_size + num_new_tokens
            if vocab_size != len(self.tokenizer):
                print(f"### [Aviso] Incompatibilidade detectada: vocab_size recalculado={vocab_size}, len(tokenizer)={len(self.tokenizer)}. Ajustando automaticamente.")
                vocab_size = len(self.tokenizer)

            model.resize_token_embeddings(vocab_size)
            with torch.no_grad():
                input_embeddings = model.get_input_embeddings().weight
                output_embeddings = model.get_output_embeddings().weight
                input_embeddings[-num_new_tokens:].copy_(
                    input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                )
                output_embeddings[-num_new_tokens:].copy_(
                    output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                )

        if freeze_params:
            print('### Freeze LLM', flush=True)
            for name, param in model.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad_(False)

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

        rebind=True is optional zero-copy loading for newly constructed models
        BEFORE optimizer creation. Rebinding storages may change dtype and
        affect models with shared but distinct parameter objects. Keep it off
        for training unless weight aliasing is reviewed.
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
                'loaded': len(seen), 'shards': len(shard_paths)}

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
        rebind = (config.get('gpu_rebind_weights', False) if gpu_run
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
            if report['unexpected']:
                print('### Unexpected checkpoint keys:',
                      report['unexpected'][:12], flush=True)
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
        missing = set(missing_keys)
        self.init_params = [name for name in self.init_params if name in missing]

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
        if vision.device != target:
            vision = vision.to(target, non_blocking=True)
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
                vision_embeds=image_embeds.to(bridge_device, non_blocking=True),
                vision_atts=image_atts.to(bridge_device, non_blocking=True),
            )

        video_v2t_feats, video_v2t_atts = None, None
        if video is not None:
            video_v2t_feats, video_v2t_atts = self.bridge(
                vision_embeds=video_embeds.to(bridge_device, non_blocking=True),
                vision_atts=video_atts.to(bridge_device, non_blocking=True),
            )

        def _concate(video, image):
            assert ((video is None) and (image is None)) is False
            if video is None:
                return image
            elif image is None:
                return video
            else:
                return torch.concat([video, image], dim=0)  # video first

        return _concate(video_v2t_feats, img_v2t_feats), _concate(video_v2t_atts, img_v2t_atts)

    @torch.no_grad()
    def generate(self, vision_input, input_ids, input_atts,
                 use_nucleus_sampling=False, num_beams=5, max_new_tokens=64, min_length=2, top_p=0.9, top_k=3,
                 repetition_penalty=1.0, no_repeat_ngram_size=3,
                 length_penalty=1.0, num_return_sequences=1, temperature=1, apply_lemmatizer=False):
        
        # Keep tensors on the embedding model's device. The original code
        # needlessly transferred embeddings CUDA -> CPU before generate().
        embed_device = self.LLM.model.embed_tokens.weight.device
        input_ids = input_ids.to(embed_device, non_blocking=True)
        text_embeds = self.embed_tokens(input_ids)
        target_device = text_embeds.device
        input_atts = input_atts.to(target_device, non_blocking=True)

        if vision_input is not None:
            # Works when the vision encoder / bridge are on different devices.
            vision_device = self._module_device(
                self.vision_encoder, getattr(self, 'vision_device', vision_input.device)
            )
            vision_input = vision_input.to(vision_device, non_blocking=True)
            vision_embeds, vision_atts = self.get_vision_embeds(vision_input)
            bridge_device = self._module_device(
                self.bridge, getattr(self, 'vision_device', vision_device)
            )
            vision_embeds = vision_embeds.to(bridge_device, non_blocking=True)
            vision_atts = vision_atts.to(bridge_device, non_blocking=True)
            v2t_feats, v2t_atts = self.bridge(
                vision_embeds=vision_embeds, vision_atts=vision_atts
            )

            v2t_feats = v2t_feats.to(
                device=target_device, dtype=text_embeds.dtype, non_blocking=True
            )
            v2t_atts = v2t_atts.to(target_device, non_blocking=True)

            inputs_embeds = torch.cat((v2t_feats, text_embeds), dim=1)
            attention_mask = torch.cat((v2t_atts, input_atts), dim=1)
        else:
            inputs_embeds = text_embeds
            attention_mask = input_atts

        outputs = self.LLM.generate(
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
        )

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

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
