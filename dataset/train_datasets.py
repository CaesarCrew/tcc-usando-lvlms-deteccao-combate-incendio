import os
import io

from base64 import b64decode
import json
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
import hashlib

import torch
from torch.utils.data import Dataset

from PIL import Image

from dataset.utils import sample_frame_ids
from dataset import get_image_transform, get_video_transform
from dataset.tokenizers import build_tokenizer

class MmapAnnotations:
    def __init__(self, parquet_path):
        self._table = pq.read_table(parquet_path, memory_map=True)
        self._columns = self._table.column_names
 
    def __len__(self):
        return self._table.num_rows
 
    def __getitem__(self, idx):
        return {col: self._table.column(col)[idx].as_py() for col in self._columns}

class LynxTrainDatasetBase(Dataset):
    def __init__(self, config):

        self.VISION_PROMPT_DICT = config.get("vision_prompt_dict", "image").strip()
        self.OUTPUT_PROMPT_DICT = config.get("output_prompt_dict", "answer").strip()

        self.prompt = config.get("prompt", "").strip()

        if len(self.prompt):
            print("### Use prompt: ", self.prompt, flush=True)

        self.data = self.read_ann_file(config['train_files'])
        cache_path = _cache_path_for(
            config['train_files'], cache_dir=config.get("ann_cache_dir", "./ann_cache")
        )
        if not os.path.exists(cache_path):
            pq.write_table(pa.Table.from_pylist(self.data), cache_path)
        self.data = MmapAnnotations(cache_path)
        
        print("### num_test: ", len(self.data), flush=True)

        self.image_rdir = config['image_rdir']

        self.use_left_pad = config['use_left_pad']
        self.tokenizer, _ = build_tokenizer(config['LLM'], use_left_pad=self.use_left_pad)
        self.max_input_tokens = config['max_input_tokens']

        self.lower_text = config['lower_text']

        _, self.img_transform = get_image_transform(config)

        self.num_frames = config['data']['num_frames']
        print("### num_frames: ", self.num_frames, flush=True)

        _, self.video_transform = get_video_transform(config)

    def read_ann_file(self, ann_file):
        if isinstance(ann_file, str):
            ann_file = [ann_file]
        else:
            assert isinstance(ann_file, list)

        data = []
        for rpath in ann_file:
            assert os.path.exists(rpath)

            if rpath.endswith(".jsonl"):
                with open(rpath, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                raise ValueError("type of rpath not json/jsonl: ", rpath)

        return data

    def __len__(self):
        return len(self.data)

    def _get_frames(self, vision_inp):
        assert isinstance(vision_inp, list) or isinstance(vision_inp, np.ndarray)
        assert isinstance(vision_inp[0], str)

        # sample frames
        assert self.num_frames > 0
        ids = sample_frame_ids(len(vision_inp), self.num_frames, training=True)

        frames = []
        for i in ids:
            img = Image.open(io.BytesIO(b64decode(vision_inp[i]))).convert("RGB")
            frames.append(self.video_transform(img))

        return frames

    def load_vision_inp(self, vision_inp):
        if vision_inp is None:
            return None

        elif isinstance(vision_inp, list) or isinstance(vision_inp, np.ndarray):
            return self._get_frames(vision_inp)

        elif isinstance(vision_inp, str):

            if os.path.exists(vision_inp):
                if vision_inp.endswith('.json'):
                    with open(vision_inp, 'r') as f:
                        data = json.load(f)

                        if isinstance(data, list) or isinstance(data, np.ndarray):
                            return self._get_frames(data)
                        else:
                            image = Image.open(io.BytesIO(b64decode(data))).convert("RGB")

                else:
                    image = Image.open(vision_inp).convert('RGB')

            else:  # base64 encoding
                try:
                    image = Image.open(io.BytesIO(b64decode(vision_inp))).convert("RGB")
                except Exception as e:
                    raise ValueError(f"check whether it is a rpath (and not exist)?: {vision_inp} {e}")

        else:
            raise ValueError(f"vision_inp type of {type(vision_inp)}")

        image = self.img_transform(image)

        return image

    def get_vision_input(self, ann):
        assert isinstance(ann, dict)

        vision_key = self.VISION_PROMPT_DICT.strip()
        vision_inp = ann[vision_key]

        return self.load_vision_inp(vision_inp)

    def get_text_input(self, ann):
        assert isinstance(ann, dict)

        text_input = self.prompt.format_map(ann).strip()

        return text_input.lower() if self.lower_text else text_input

    def __getitem__(self, index):
        ann = self.data[index]
        assert isinstance(ann, dict)

        idx = ann['index']

        vision_input = self.get_vision_input(ann)
        text_input = self.get_text_input(ann)

        input_ids = [self.tokenizer.bos_token] + self.tokenizer.tokenize(text_input)
        if self.max_input_tokens > 0:
            input_ids = input_ids[-self.max_input_tokens:]

        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)

        return idx, vision_input, input_ids

    def pad_batch_data(self, input_ids):
        max_length = max([len(ids) for ids in input_ids])

        # pad + get atts
        input_ids_pad = []
        input_atts_pad = []

        for ids in input_ids:
            n_tokens = len(ids)
            n_pads = max_length - n_tokens

            if self.use_left_pad:  # recommend
                input_ids_pad.append([self.tokenizer.pad_token_id] * n_pads + ids)
                input_atts_pad.append([0] * n_pads + [1] * n_tokens)

            else:  # right pad
                input_ids_pad.append(ids + [self.tokenizer.pad_token_id] * n_pads)
                input_atts_pad.append([1] * n_tokens + [0] * n_pads)

        input_ids_pad = torch.LongTensor(input_ids_pad)
        input_atts_pad = torch.LongTensor(input_atts_pad)

        return input_ids_pad, input_atts_pad

    def collate_fn(self, batch):
        inputs = []
        for x in zip(*batch):
            inputs.append(x)

        idx, vision_input, input_ids = inputs

        if isinstance(vision_input[0], list):  # video
            batch_size = len(vision_input)
            vision_input = torch.stack(sum(vision_input, []))  # flatten
            _, c, h, w = vision_input.shape

            if self.num_frames > 1:
                vision_input = vision_input.reshape([batch_size, self.num_frames, c, h, w])

        else:  # image
            vision_input = torch.stack(vision_input, dim=0)

        input_ids_pad, input_atts_pad = self.pad_batch_data(input_ids)

        return idx, vision_input, input_ids_pad, input_atts_pad

IGNORE_TOKEN_ID = -100

class LynxTrainDataset(Dataset):
    def __init__(self, eval_dataset, indices, max_train_tokens=256):
        self.eval_dataset = eval_dataset
        self.indices = list(indices)
        self.max_train_tokens = max_train_tokens

        self.output_key = eval_dataset.OUTPUT_PROMPT_DICT.strip()
        self.lower_text = eval_dataset.lower_text
        self.use_left_pad = eval_dataset.use_left_pad
        self.tokenizer = eval_dataset.tokenizer

    def __len__(self):
        return len(self.indices)

    def _build_text_tokens(self, ann):
        prompt_text = self.eval_dataset.get_text_input(ann)
        answer_text = str(ann[self.output_key]).strip()
        if self.lower_text:
            answer_text = answer_text.lower()

        prompt_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(" " + answer_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        if len(answer_ids) + 1 >= self.max_train_tokens:
            answer_ids = answer_ids[-(self.max_train_tokens - 1):]
            prompt_ids = [self.tokenizer.bos_token_id]

        keep_prompt = self.max_train_tokens - len(answer_ids)
        prompt_ids = [prompt_ids[0]] + prompt_ids[-(keep_prompt - 1):]

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids

        return input_ids, labels

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        ann = self.eval_dataset.data[data_idx]

        vision_input = self.eval_dataset.get_vision_input(ann)
        input_ids, labels = self._build_text_tokens(ann)

        return ann["index"], vision_input, input_ids, labels

    def collate_fn(self, batch):
        idx, vision_input, input_ids, labels = zip(*batch)

        if isinstance(vision_input[0], list):
            batch_size = len(vision_input)
            vision_input = torch.stack(sum(vision_input, []))
            _, c, h, w = vision_input.shape
            if self.eval_dataset.num_frames > 1:
                vision_input = vision_input.reshape([batch_size, self.eval_dataset.num_frames, c, h, w])
        else:
            vision_input = torch.stack(vision_input, dim=0)

        max_len = max(len(x) for x in input_ids)

        input_ids_pad = []
        input_atts_pad = []
        labels_pad = []
        for ids, labs in zip(input_ids, labels):
            n_tokens = len(ids)
            n_pads = max_len - n_tokens

            if self.use_left_pad:
                input_ids_pad.append([self.tokenizer.pad_token_id] * n_pads + ids)
                input_atts_pad.append([0] * n_pads + [1] * n_tokens)
                labels_pad.append([-100] * n_pads + labs)
            else:
                input_ids_pad.append(ids + [self.tokenizer.pad_token_id] * n_pads)
                input_atts_pad.append([1] * n_tokens + [0] * n_pads)
                labels_pad.append(labs + [-100] * n_pads)

        return (
            idx,
            vision_input,
            torch.LongTensor(input_ids_pad),
            torch.LongTensor(input_atts_pad),
            torch.LongTensor(labels_pad),
        )

class TrainFoldDataset(Dataset):
    def __init__(self, base_dataset, indices, max_total_tokens=128):
        self.base_dataset = base_dataset
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

    def _build_text_tokens(self, ann):
        prompt_text = self.base_dataset.get_text_input(ann)
        answer_text = str(ann[self.output_key]).strip()
        if self.lower_text:
            answer_text = answer_text.lower()

        prompt_tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(prompt_text)
        answer_tokens = self.tokenizer.tokenize(" " + answer_text) + [self.tokenizer.eos_token]

        return prompt_tokens, answer_tokens

    def _get_text_ids(self, ann):
        prompt_tokens, answer_tokens = self._build_text_tokens(ann)

        prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)

        prompt_ids, answer_ids = self._truncate(prompt_ids, answer_ids)

        input_ids = prompt_ids + answer_ids
        labels = ([IGNORE_TOKEN_ID] * len(prompt_ids)) + answer_ids

        return input_ids, labels

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        ann = self.base_dataset.data[data_idx]

        if self.output_key not in ann:
            raise KeyError(f"Missing key '{self.output_key}' in sample index {idx}")

        vision_input = self.base_dataset.get_vision_input(ann)
        input_ids, labels = self._get_text_ids(ann)

        return data_idx, vision_input, input_ids, labels #, ann["index"]

    def _pad_sequences(self, input_ids, labels):
        max_len = max(len(x) for x in input_ids)
        if self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
        else:
            pad_id = self.tokenizer.eos_token_id

        input_ids_pad = []
        input_atts_pad = []
        labels_pad = []

        assert len(input_ids) == len(labels), "input_ids and labels must have the same length"

        for ids, lbs in zip(input_ids, labels):
            n = len(ids)
            n_pad = max_len - n

            if self.use_left_pad:
                input_ids_pad.append(([pad_id] * n_pad) + ids)
                input_atts_pad.append(([0] * n_pad) + ([1] * n))
                labels_pad.append(([IGNORE_TOKEN_ID] * n_pad) + lbs)
            else:
                input_ids_pad.append(ids + ([pad_id] * n_pad))
                input_atts_pad.append(([1] * n) + ([0] * n_pad))
                labels_pad.append(lbs + ([IGNORE_TOKEN_ID] * n_pad))

        return (
            torch.LongTensor(input_ids_pad),
            torch.LongTensor(input_atts_pad),
            torch.LongTensor(labels_pad)
        )

    def collate_fn(self, batch):
        idx, vision_input, input_ids, labels = zip(*batch)

        if isinstance(vision_input[0], list):
            batch_size = len(vision_input)
            vision_input = torch.stack(sum(vision_input, []))
            _, c, h, w = vision_input.shape
            if self.base_dataset.num_frames > 1:
                vision_input = vision_input.reshape([batch_size, self.base_dataset.num_frames, c, h, w])
        else:
            vision_input = torch.stack(vision_input, dim=0)

        input_ids_pad, input_atts_pad, labels_pad = self._pad_sequences(input_ids, labels)
        return idx, vision_input, input_ids_pad, input_atts_pad, labels_pad

    def eval_collate_fn(self, batch):
        idx, vision_input, input_ids, labels = zip(*batch)

        if isinstance(vision_input[0], list):
            batch_size = len(vision_input)
            vision_input = torch.stack(sum(vision_input, []))
            _, c, h, w = vision_input.shape
            if self.base_dataset.num_frames > 1:
                vision_input = vision_input.reshape([batch_size, self.base_dataset.num_frames, c, h, w])
        else:
            vision_input = torch.stack(vision_input, dim=0)

        input_ids_pad, input_atts_pad = self._pad_sequences(input_ids, labels)
        return idx, vision_input, input_ids_pad, input_atts_pad

def _cache_path_for(train_files, cache_dir):
    paths = [train_files] if isinstance(train_files, str) else list(train_files)
    digest = hashlib.sha1("|".join(sorted(paths)).encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"annotations_{digest}.parquet")