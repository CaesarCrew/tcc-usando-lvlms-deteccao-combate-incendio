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

import argparse
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
import re
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

import utils
from utils import write_jsonl
from dataset import create_dataset, create_loader

torch.set_default_dtype(torch.float16)
@ torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    result = []

    for n, (idx, vision_input, input_ids, input_atts) in enumerate(data_loader):
        vision_input = vision_input.to(device, non_blocking=True)
        input_ids = input_ids[:, -128:]
        #input_ids = input_ids.to(device)
        input_atts = input_atts[:, -128:]
        input_atts = input_atts.to(device)#.half()

        with torch.amp.autocast('cuda'):
            text_outputs = model.generate(
                vision_input=vision_input,
                input_ids=input_ids, input_atts=input_atts,
                use_nucleus_sampling=config.get('use_nucleus_sampling', False),
                apply_lemmatizer=config['apply_lemmatizer'],
                num_beams=config['num_beams'],
                min_length=config['min_length'],
                length_penalty=config.get('length_penalty', 1.0),
                no_repeat_ngram_size=config.get('no_repeat_ngram_size', -1),
                top_p=config.get('top_p', 0.9),
                top_k=config.get('top_k', 3),
                max_new_tokens=config.get('max_new_tokens', 64))

        for i, output in zip(idx, text_outputs):
            result.append({"index": i, "text_output": output.strip()})
        torch.cuda.empty_cache()

    return result

def make_k_folds(n, k, seed=123, shuffle=True):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    folds = np.array_split(indices, k)  # nearly equal sizes
    return folds


def extract_binary_class_from_rpath(rpath):
    parts = Path(rpath).parts
    if "nofire" in parts:
        return "nofire"
    if "fire" in parts:
        return "fire"
    raise ValueError(f"Could not infer class from rpath: {rpath}")


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

    f1_fire = class_f1("fire")
    f1_nofire = class_f1("nofire")

    return {
        "accuracy": accuracy,
        "f1_fire": f1_fire,
        "f1_nofire": f1_nofire,
        "f1_macro": (f1_fire + f1_nofire) / 2 if total else 0.0,
    }

def main(args, config):
    print("### Evaluating", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("config:", json.dumps(config), flush=True)
    print("output_path, ", args.output_path, flush=True)

    print("### Creating model", flush=True)
    from models.lynx import LynxBase
    model = LynxBase(config=config, freeze_vit=config['freeze_vit'], freeze_llm=config['freeze_llm'], load_bridge=False)
    model.vision_encoder = model.vision_encoder.to(device).half()
    model.bridge = model.bridge.to(device).half()
    model.LLM = model.LLM.to("cpu")#.float()

    for _, param in model.named_parameters():
        param.requires_grad = False

    model.eval()

    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("### Creating datasets", flush=True)
    test_dataset = create_dataset('eval', config)

    print("### Setting up k-fold cross-validation", flush=True)
    n = len(test_dataset)
    k = 5
    assert k >= 2, "k_folds must be >= 2"
    assert k <= n, f"k_folds must be <= dataset size (n={n})"

    folds = make_k_folds(n=n, k=k, seed=args.fold_seed, shuffle=True)

    start_time = time.time()
    print(f"### Start {k}-fold evaluating (outcome CV)", flush=True)

    all_predictions = []
    per_fold_paths = []
    per_fold_times = []

    for fold_i, fold_indices in enumerate(folds):
        print(f"### Fold {fold_i+1}/{k}: n={len(fold_indices)}", flush=True)

        fold_ds = Subset(test_dataset, fold_indices.tolist())
        fold_annotations = [test_dataset.data[i] for i in fold_indices.tolist()]

        fold_loader = create_loader([fold_ds],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    collate_fns=[test_dataset.collate_fn])[0]
        
        start_time_fold = time.time()
        fold_preds = evaluation(model, fold_loader, device, config)
        fold_time = time.time() - start_time_fold
        fold_time_str = str(datetime.timedelta(seconds=int(fold_time)))
        per_fold_times.append(fold_time_str)

        fold_targets = []
        fold_predictions = []
        for ann, pred in zip(fold_annotations, fold_preds):
            target_class = extract_binary_class_from_rpath(ann["image"])
            predicted_class = predict_binary_class(pred["text_output"])
            fold_targets.append(target_class)
            fold_predictions.append(predicted_class)
            pred["target_class"] = target_class
            pred["predicted_class"] = predicted_class
            pred["image_path"] = ann["image"]

        fold_metrics = compute_binary_metrics(fold_targets, fold_predictions)
        print(
            "### Fold {}/{} metrics: accuracy={:.4f}, f1_fire={:.4f}, f1_nofire={:.4f}, f1_macro={:.4f}".format(
                fold_i + 1,
                k,
                fold_metrics["accuracy"],
                fold_metrics["f1_fire"],
                fold_metrics["f1_nofire"],
                fold_metrics["f1_macro"],
            ),
            flush=True,
        )

        # tag fold id
        for p in fold_preds:
            p["fold"] = fold_i

        all_predictions.extend(fold_preds)

        if args.save_per_fold:
            fold_out = args.output_path.replace(".jsonl", f"_fold{fold_i}.jsonl")
            write_jsonl(fold_preds, fold_out)
            per_fold_paths.append(fold_out)
            print("### Fold results saved to:", fold_out, flush=True)

        # To prevent GPU memory issues, we can clear the model and empty cache after each fold
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # Save combined
    write_jsonl(all_predictions, args.output_path)
    print("### Combined prediction results saved to:", args.output_path, flush=True)

    print('### Time {}'.format(total_time_str))
    print('### Times per fold {}'.format(per_fold_times))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True, help="path of outputfile")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--k_folds', default=5, type=int)
    parser.add_argument('--fold_seed', default=123, type=int)
    parser.add_argument('--save_per_fold', action='store_true')

    args = parser.parse_args()

    yaml = YAML(typ='rt')
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    main(args, config)
