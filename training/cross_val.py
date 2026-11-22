import argparse
import copy
import datetime as dt
from datetime import datetime
import json
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset, Subset

from calculate_metrics import process_metrics
from dataset import create_train_dataset
from dataset.train_datasets import TrainFoldDataset
from tcc.utils import get_rank, write_jsonl
from training.utils import resolve_data_files


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


def build_model(config, device, llm_device="cpu", train_adapters=False):
    from models.lynx import LynxBase

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        cuda_memory_report("before LynxBase ctor")

    model = LynxBase(
        config=config,
        freeze_vit=config["freeze_vit"],
        freeze_llm=config["freeze_llm"],
        load_bridge=False,
    )

    cuda_memory_report("after LynxBase ctor")

    if device.type == "cuda":
        model.vision_encoder = model.vision_encoder.to(device).half()
        model.bridge = model.bridge.to(device).half()
    else:
        model.vision_encoder = model.vision_encoder.to(device)
        model.bridge = model.bridge.to(device)

    cuda_memory_report("after vision_encoder + bridge placement")

    if llm_device != "cpu":
        print("### For cross-validation, keeping LLM on CPU to match generate.py placement", flush=True)
    
    llm_device = torch.device("cpu")
    model.LLM = model.LLM.to(llm_device)

    cuda_memory_report("after LLM placement")

    for _, p in model.named_parameters():
        p.requires_grad = False

    for _, p in model.bridge.named_parameters():
        p.requires_grad = True

    if train_adapters:
        for n, p in model.LLM.named_parameters():
            if "adapter" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    cuda_memory_report("after freezing + eval")

    return model


def forward_train_loss(model, batch, device):
    _, vision_input, input_ids, input_atts, labels = batch

    llm_device = next(model.LLM.parameters()).device
    input_ids = input_ids.to(llm_device)
    input_atts = input_atts.to(llm_device)
    labels = labels.to(llm_device)

    text_embeds = model.embed_tokens(input_ids)

    if vision_input is not None:
        vision_input = vision_input.to(device, non_blocking=True)
        with torch.set_grad_enabled(not model.freeze_vit):
            vision_embeds, vision_atts = model.get_vision_embeds(vision_input)

        v2t_feats, v2t_atts = model.bridge(vision_embeds=vision_embeds, vision_atts=vision_atts)
        v2t_feats = v2t_feats.to(llm_device, dtype=text_embeds.dtype)
        v2t_atts = v2t_atts.to(llm_device)

        v2t_ignore_labels = torch.full(
            (labels.shape[0], v2t_feats.shape[1]),
            -100,
            dtype=labels.dtype,
            device=llm_device,
        )

        inputs_embeds = torch.cat([v2t_feats, text_embeds], dim=1)
        attention_mask = torch.cat([v2t_atts, input_atts], dim=1)
        labels = torch.cat([v2t_ignore_labels, labels], dim=1)
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


def train_one_fold(model, train_loader, device, epochs, lr, weight_decay, grad_accum_steps, grad_clip):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    model.train()
    model.vision_encoder.eval()

    history = []
    global_step = 0

    for epoch in range(epochs):
        running_loss = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader):
            loss = forward_train_loss(model, batch, device)
            loss = loss / grad_accum_steps
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()
                global_step += 1

            running_loss += loss.item() * grad_accum_steps
            n_steps += 1

        if n_steps % grad_accum_steps != 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            global_step += 1

        epoch_loss = running_loss / max(1, n_steps)
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "global_step": global_step})
        print(f"### Finishing epoch {epoch} and aplying gradients from Step {step + 1}/{len(train_loader)}", flush=True)

    return history


def evaluate_one_fold(model, base_dataset, val_indices, device, config, num_workers):
    val_subset = Subset(base_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size_train,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=base_dataset.collate_fn,
        drop_last=False,
    )

    preds = []
    model.eval()
    with torch.inference_mode():
        for idx, vision_input, input_ids, input_atts in val_loader:
            vision_input = vision_input.to(device, non_blocking=True)
            input_ids = input_ids[:, -128:].to(device)
            input_atts = input_atts[:, -128:].to("cuda", non_blocking=True)

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
                max_new_tokens=config.get("max_new_tokens", 64),
            )

            for i, output in zip(idx, text_outputs):
                preds.append({"index": i, "text_output": output.strip()})

            torch.cuda.empty_cache()

    annotations = [base_dataset.data[i] for i in val_indices]
    metrics = process_metrics(annotations, preds)
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_num_threads(max(1, min(12, (os.cpu_count() or 1))))
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    print(f"### Number of CPU threads: {torch.get_num_threads()}", flush=True)

    seed = args.seed + get_rank()
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    cudnn.benchmark = True

    data_files = resolve_data_files(args, config)
    config_for_data = copy.deepcopy(config)
    config_for_data["test_files"] = data_files

    print("### Creating dataset for cross-validation", flush=True)
    full_dataset = create_train_dataset("eval", config_for_data)
    n = len(full_dataset)
    assert args.k_folds >= 2, "k_folds must be >= 2"
    assert args.k_folds <= n, f"k_folds must be <= dataset size (n={n})"

    folds = make_kfold_indices(
        full_dataset.data,
        k=args.k_folds,
        seed=args.fold_seed,
        stratified=(not args.disable_stratified)
    )

    print("### Fold sizes:", [len(f) for f in folds], flush=True)

    all_predictions = []
    all_fold_metrics = []
    fold_reports = []
    num_workers = max(2, min(8, (os.cpu_count() or 1) // 2))
    print(f"### Using {num_workers} workers for DataLoader", flush=True)

    print(f"### Start {args.k_folds}-fold cross-validation with retraining", flush=True)
    start_cv = time.time()

    for fold_i, val_indices in enumerate(folds):
        fold_no = fold_i + 1
        #val_set = set(val_indices)
        #train_indices = [i for i in range(n) if i not in val_set]
        train_indices = sorted(list(set(range(n)) - set(val_indices)))

        print(
            f"### Fold {fold_no}/{args.k_folds}: train={len(train_indices)} val={len(val_indices)}",
            flush=True,
        )

        if device == torch.device("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            cuda_memory_report(f"fold {fold_no} before model build")

        model = build_model(
            config=config,
            device=device,
            llm_device=args.llm_device,
            train_adapters=args.train_adapters
        )

        train_dataset = TrainFoldDataset(
            base_dataset=full_dataset,
            indices=train_indices,
            max_total_tokens=args.max_train_tokens,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size_train,
            num_workers=args.num_workers if args.num_workers > 0 else num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            drop_last=False,
        )

        if device == torch.device("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
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

        del model
        torch.cuda.empty_cache()

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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm_device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--k_folds", type=int, default=5, required=True)
    parser.add_argument("--fold_seed", type=int, default=123)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5) #2e-5
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_tokens", type=int, default=256) # 128
    parser.add_argument("--num_workers", type=int, default=4)

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