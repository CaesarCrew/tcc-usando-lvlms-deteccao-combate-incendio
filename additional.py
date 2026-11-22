import re
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


import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import InterpolationMode

import torch
import torch.distributed as dist

IGNORE_TOKEN_ID = -100

def create_train_dataset(dataset, config):
    if dataset == 'eval':
        train_dataset = LynxTrainDatasetBase(config)
        return train_dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")

def resolve_data_files(args, config):
    if "train_files" in config:
        return config["train_files"]
    return config["test_files"]

def extract_class(answer):
    if answer == "yes":
        return "fire"
    if answer == "no":
        return "nofire"

def predict_binary_class(text_output):
    text = text_output.strip().lower()
    lead_text = text.lstrip(", ")

    if re.match(r"^(no|nope|nah)\b", lead_text):
        return "nofire"
    if re.match(r"^yes\b", lead_text):
        return "fire"

    nofire_patterns = [
        r"\bno fire\b",
        r"\bnofire\b",
        r"\bno flames?\b",
        r"\bno burning\b",
        r"\bno wildfire\b",
        r"\bno signs? of fire\b",
        r"\bnot a fire\b",
    ]
    fire_patterns = [
        r"\byes\b",
        r"\bfire\b",
        r"\bflames?\b",
        r"\bwildfire\b",
        r"\bblaze\b",
        r"\bburning\b",
        r"\bsmoke\b",
    ]

    if any(re.search(pattern, text) for pattern in nofire_patterns):
        return "nofire"
    if any(re.search(pattern, text) for pattern in fire_patterns):
        return "fire"

    # Fall back to the first token so short answers like "no"/"yes" still work.
    first_token = re.sub(r"^[^a-z]+|[^a-z]+$", "", lead_text.split(maxsplit=1)[0]) if text else ""
    if first_token in {"no", "nope"}:
        return "nofire"
    else:
        return "fire"
    
    return "no answer"

def get_image_transform(config):

    normalize = transforms.Normalize(config['image_mean'], config['image_std'])

    def _convert_to_rgb(image):
        return image.convert('RGB')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(config['image_res'], config['image_res']), scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                                     interpolation=InterpolationMode.BICUBIC),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=config['image_res'], interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=(config['image_res'], config['image_res'])),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


def get_video_transform(config):
    normalize = transforms.Normalize(config['image_mean'], config['image_std'])

    def _convert_to_rgb(image):
        return image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(size=config['image_res'], interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=(config['image_res'], config['image_res'])),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ])

    return transform, transform

def compute_binary_metrics(targets, predictions):
    assert len(targets) == len(predictions)

    total = len(targets)
    correct = sum(t == p for t, p in zip(targets, predictions))
    accuracy = correct / total if total else 0.0

    def class_f1(positive_class):
        tp = sum((t == positive_class) and (p == positive_class) for t, p in zip(targets, predictions))
        fp = sum((t != positive_class) and (p == positive_class) for t, p in zip(targets, predictions))
        fn = sum((t == positive_class) and (p != positive_class) for t, p in zip(targets, predictions))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    
    def class_mcc(positive_class):
        tp = sum((t == positive_class) and (p == positive_class) for t, p in zip(targets, predictions))
        if(positive_class == "fire"):
            tn = sum((t == "nofire") and (p == "nofire") for t, p in zip(targets, predictions))
            fp = sum((t == "nofire") and (p == positive_class) for t, p in zip(targets, predictions))
            fn = sum((t == positive_class) and (p == "nofire") for t, p in zip(targets, predictions))
        if(positive_class == "nofire"):
            tn = sum((t == "fire") and (p == "fire") for t, p in zip(targets, predictions))
            fp = sum((t == "fire") and (p == positive_class) for t, p in zip(targets, predictions))
            fn = sum((t == positive_class) and (p == "fire") for t, p in zip(targets, predictions))
        
        #print(f"Fold tn: {tn} Fold fp: {fp}")
        nominator = (tp * tn) - (fp * fn)
        normalized_denominator = pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5)
        if normalized_denominator != 0:
            return nominator / normalized_denominator
        else:
            print(f"### No images of class {positive_class}")
            return 0

    f1_fire = class_f1("fire")
    f1_nofire = class_f1("nofire")
    mcc_fire = class_mcc("fire")
    mcc_nofire = class_mcc("nofire")

    return {
        "accuracy": accuracy,
        "f1_fire": f1_fire,
        "f1_nofire": f1_nofire,
        "f1_macro": (f1_fire + f1_nofire) / 2 if total else 0.0,
        "mcc_fire": mcc_fire,
        "mcc_nofire": mcc_nofire,
    }

def process_metrics(annotations, predictions):
    targets = []
    predictions_format = []
    for ann, pred in zip(annotations, predictions):
        target_class = extract_class(ann["answer"][0])
        predicted_class = predict_binary_class(pred["text_output"])
        targets.append(target_class)
        predictions_format.append(predicted_class)
        pred["target_class"] = target_class
        pred["predicted_class"] = predicted_class
        pred["image_path"] = ann["image"]
    
    #print(f"Targets: {targets} \n")
    #print(f"Predictions: {predictions_format} \n")
    metrics = compute_binary_metrics(targets, predictions_format)
    return metrics

def sample_frame_ids(num_frames: int, num_segments: int, training: bool):
    nf = num_frames
    ns = num_segments
    out_indices = np.zeros(ns)
    if nf <= ns:
        if training:
            out_indices = np.concatenate((np.arange(nf), np.random.randint(nf, size=ns - nf)), axis=0)
        else:
            out_indices = np.array([(idx % nf) for idx in range(ns)])
        out_indices = np.sort(np.array(out_indices))
    else:
        stride = nf // ns  # at least 1
        strides = np.array([stride] * ns)
        offsets = np.array([1] * (nf - ns * stride) + [0] * (ns * (stride + 1) - nf))
        if training:
            np.random.shuffle(offsets)
        strides += offsets
        cursor = 0
        for idx, each_stride in enumerate(strides):
            left, right = cursor, cursor + each_stride
            cursor += each_stride
            if training:
                out_indices[idx] = np.random.randint(left, right)
            else:
                out_indices[idx] = left

    return [int(i) for i in out_indices]

DEFAULT_PAD_TOKEN = "[PAD]"
TOKEN_NONE_FLAG = "[NONE]"

def build_tokenizer(LLM: str, use_left_pad: bool):
    rpath = os.path.join("data/", LLM)
    assert os.path.exists(rpath)

    num_new_tokens = 0

    if LLM in ['vicuna-7b']:
        from models.llms.llama.tokenization_llama import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(rpath)

        if tokenizer.pad_token is None:
            num_new_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": DEFAULT_PAD_TOKEN,
                }
            )

    else:
        raise NotImplementedError("Tokenizer for LLM: ", LLM)

    print("-" * 40)
    print("### Vocab Size: ", len(tokenizer), flush=True)

    assert tokenizer.eos_token is not None
    assert tokenizer.pad_token is not None

    if tokenizer.bos_token is None:
        print("set bos_token to: ", TOKEN_NONE_FLAG, flush=True)
        tokenizer.bos_token = TOKEN_NONE_FLAG

    else:
        print("bos_token, ", tokenizer.bos_token)
        print("bos_token_id, ", tokenizer.bos_token_id)

    if use_left_pad:
        tokenizer.padding_side = "left"

    print("Left Pad: ", use_left_pad, flush=True)

    print("eos_token, ", tokenizer.eos_token)
    print("eos_token_id, ", tokenizer.eos_token_id)

    print("pad_token, ", tokenizer.pad_token)
    print("pad_token_id, ", tokenizer.pad_token_id)

    print("unk_token, ", tokenizer.unk_token, flush=True)
    print("unk_token_id, ", tokenizer.unk_token_id, flush=True)
    print("-" * 40)

    return tokenizer, num_new_tokens


def write_jsonl(data, wpath, option):
    with open(wpath, option) as f:
        for sample in data:
            f.write(json.dumps(sample)+'\n')


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def _cache_path_for(train_files, cache_dir):
    paths = [train_files] if isinstance(train_files, str) else list(train_files)
    digest = hashlib.sha1("|".join(sorted(paths)).encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"annotations_{digest}.parquet")

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