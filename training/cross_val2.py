import copy
import datetime as dt
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset, Subset

from calculate_metrics import extract_class, process_metrics
from dataset import create_dataset
from utils import get_rank, write_jsonl


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


def make_stratified_k_folds(annotations, k, seed=123):
    by_class = {}
    for i, ann in enumerate(annotations):
        cls = extract_class(str(ann["answer"]).strip().lower())
        by_class.setdefault(cls, []).append(i)

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls_indices in by_class.values():
        cls_indices = np.array(cls_indices)
        rng.shuffle(cls_indices)
        for pos, idx in enumerate(cls_indices.tolist()):
            folds[pos % k].append(idx)

    for fold in folds:
        fold.sort()

    return folds


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


def get_trainable_params(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    return trainable, n_params


def forward_lm(model, vision_input, input_ids, input_atts, labels, llm_device):
    input_ids = input_ids.to(llm_device, non_blocking=True)
    input_atts = input_atts.to(llm_device, non_blocking=True)
    labels = labels.to(llm_device, non_blocking=True)

    text_embeds = model.embed_tokens(input_ids)

    if vision_input is not None:
        vision_embeds, vision_atts = model.get_vision_embeds(vision_input)
        v2t_feats, v2t_atts = model.bridge(vision_embeds=vision_embeds, vision_atts=vision_atts)

        v2t_feats = v2t_feats.to(llm_device, dtype=text_embeds.dtype)
        v2t_atts = v2t_atts.to(llm_device)

        inputs_embeds = torch.cat([v2t_feats, text_embeds], dim=1)
        attention_mask = torch.cat([v2t_atts, input_atts], dim=1)

        ignore_prefix = torch.full(
            (labels.size(0), v2t_feats.size(1)),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        labels = torch.cat([ignore_prefix, labels], dim=1)
    else:
        inputs_embeds = text_embeds
        attention_mask = input_atts

    outputs = model.LLM(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )

    return outputs.loss


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    vision_device,
    llm_device,
    epoch,
    epochs,
    grad_clip=1.0,
):
    model.train()
    running_loss = 0.0

    for step, (_, vision_input, input_ids, input_atts, labels) in enumerate(train_loader, start=1):
        vision_input = vision_input.to(vision_device, non_blocking=True)

        loss = forward_lm(model, vision_input, input_ids, input_atts, labels, llm_device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=grad_clip)

        optimizer.step()

        running_loss += loss.item()
        if step % 20 == 0 or step == len(train_loader):
            avg = running_loss / step
            print(f"### Train epoch {epoch}/{epochs}, step {step}/{len(train_loader)}, avg_loss={avg:.4f}", flush=True)

    return running_loss / max(1, len(train_loader))


@torch.no_grad()
def evaluate_generation(model, data_loader, vision_device, llm_device, config):
    model.eval()
    result = []

    for idx, vision_input, input_ids, input_atts in data_loader:
        vision_input = vision_input.to(vision_device, non_blocking=True)

        input_ids = input_ids[:, -128:].to(llm_device)
        input_atts = input_atts[:, -128:].to(llm_device)

        text_embeds = model.embed_tokens(input_ids)

        vision_embeds, vision_atts = model.get_vision_embeds(vision_input)
        v2t_feats, v2t_atts = model.bridge(vision_embeds=vision_embeds, vision_atts=vision_atts)

        v2t_feats = v2t_feats.to(llm_device, dtype=text_embeds.dtype)
        v2t_atts = v2t_atts.to(llm_device)

        inputs_embeds = torch.cat([v2t_feats, text_embeds], dim=1)
        attention_mask = torch.cat([v2t_atts, input_atts], dim=1)

        outputs = model.LLM.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=config.get("use_nucleus_sampling", False),
            top_p=config.get("top_p", 0.9),
            top_k=config.get("top_k", 3),
            num_beams=config["num_beams"],
            max_new_tokens=config.get("max_new_tokens", 64),
            min_length=config["min_length"],
            length_penalty=config.get("length_penalty", 1.0),
            no_repeat_ngram_size=config.get("no_repeat_ngram_size", -1),
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )

        text_outputs = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, output in zip(idx, text_outputs):
            result.append({"index": i, "text_output": output.strip()})

    return result


def resolve_data_files(args, config):
    if args.data_files:
        return args.data_files
    if "train_files" in config:
        return config["train_files"]
    return config["test_files"]


def save_fold_checkpoint(model, optimizer, fold_id, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    cpath = os.path.join(checkpoint_dir, f"fold_{fold_id}_epoch_{epoch}.pt")
    torch.save(
        {
            "fold": fold_id,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        cpath,
    )
    print(f"### Fold {fold_id} checkpoint saved: {cpath}", flush=True)


def fold_summary_line(fold_id, metrics, train_loss, train_size, val_size, elapsed_s):
    return (
        f"Fold {fold_id}: "
        f"train={train_size}, val={val_size}, loss={train_loss:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"f1_fire={metrics['f1_fire']:.4f}, "
        f"f1_nofire={metrics['f1_nofire']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, "
        f"time={str(dt.timedelta(seconds=int(elapsed_s)))}"
    )


def main(args, config):
    vision_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.llm_device == "cuda" and torch.cuda.is_available():
        llm_device = torch.device("cuda")
    else:
        llm_device = torch.device("cpu")

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if args.unfreeze_vit:
        config["freeze_vit"] = False
    if args.unfreeze_llm:
        config["freeze_llm"] = False

    data_files = resolve_data_files(args, config)
    config_for_data = copy.deepcopy(config)
    config_for_data["test_files"] = data_files

    print("### Loading dataset for cross-validation", flush=True)
    full_dataset = create_dataset("eval", config_for_data)

    n = len(full_dataset)
    if args.k_folds < 2:
        raise ValueError("k_folds must be >= 2")
    if args.k_folds > n:
        raise ValueError(f"k_folds must be <= dataset size (n={n})")

    folds = make_stratified_k_folds(full_dataset.data, args.k_folds, args.fold_seed)

    print("### Fold sizes:", [len(f) for f in folds], flush=True)

    all_predictions = []
    fold_reports = []

    global_start = time.time()

    for fold_idx, val_indices in enumerate(folds, start=1):
        fold_start = time.time()

        train_set = set(range(n)) - set(val_indices)
        train_indices = sorted(list(train_set))

        train_dataset = LynxTrainDataset(
            full_dataset,
            train_indices,
            max_train_tokens=args.max_train_tokens,
        )
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size_train,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            collate_fn=train_dataset.collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size_test"],
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=full_dataset.collate_fn,
        )

        print(f"### Fold {fold_idx}/{args.k_folds}: building model", flush=True)
        model = build_model(config, vision_device, llm_device)

        trainable_params, n_trainable = get_trainable_params(model)
        if not trainable_params:
            raise RuntimeError("No trainable parameters. Set --unfreeze_vit and/or --unfreeze_llm if needed.")

        print(f"### Fold {fold_idx}: trainable params={n_trainable}", flush=True)

        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

        last_train_loss = 0.0
        for epoch in range(1, args.epochs + 1):
            last_train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                vision_device,
                llm_device,
                epoch,
                args.epochs,
                grad_clip=args.grad_clip,
            )

            if args.save_fold_checkpoints:
                save_fold_checkpoint(model, optimizer, fold_idx, epoch, args.checkpoint_dir)

        predictions = evaluate_generation(model, val_loader, vision_device, llm_device, config)
        val_annotations = [full_dataset.data[i] for i in val_indices]
        metrics = process_metrics(val_annotations, predictions)

        for pred in predictions:
            pred["fold"] = fold_idx

        all_predictions.extend(predictions)

        fold_elapsed = time.time() - fold_start
        fold_line = fold_summary_line(
            fold_idx,
            metrics,
            last_train_loss,
            len(train_indices),
            len(val_indices),
            fold_elapsed,
        )
        print(f"### {fold_line}", flush=True)

        fold_reports.append(
            {
                "fold": fold_idx,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "train_loss": last_train_loss,
                "metrics": metrics,
                "elapsed_seconds": fold_elapsed,
                "summary": fold_line,
            }
        )

        del model
        del optimizer
        torch.cuda.empty_cache()

    metric_names = ["accuracy", "f1_fire", "f1_nofire", "f1_macro"]
    aggregate = {}
    for name in metric_names:
        values = [fr["metrics"][name] for fr in fold_reports]
        aggregate[f"{name}_mean"] = float(np.mean(values))
        aggregate[f"{name}_std"] = float(np.std(values))

    total_time = time.time() - global_start

    if args.output_predictions:
        write_jsonl(all_predictions, args.output_predictions)
        print(f"### Saved predictions: {args.output_predictions}", flush=True)

    final_report = {
        "k_folds": args.k_folds,
        "dataset_size": n,
        "data_files": data_files,
        "epochs": args.epochs,
        "batch_size_train": args.batch_size_train,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "fold_reports": fold_reports,
        "aggregate": aggregate,
        "total_time": str(dt.timedelta(seconds=int(total_time))),
    }

    with open(args.report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"### Saved cross-validation report: {args.report_path}", flush=True)
    print("### Aggregate:", json.dumps(aggregate, indent=2), flush=True)
    print(f"### Total time: {str(dt.timedelta(seconds=int(total_time)))}", flush=True)