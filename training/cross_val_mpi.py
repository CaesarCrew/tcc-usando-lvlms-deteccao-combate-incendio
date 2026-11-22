import copy
import json

import os
from mpi4py import MPI

import time
import datetime as dt
from datetime import datetime

import numpy as np
from ruamel.yaml import YAML

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, Subset

from calculate_metrics import process_metrics
from dataset import create_train_dataset
from dataset.train_datasets import TrainFoldDataset
from training.utils import resolve_data_files, get_trainable_params, snapshot_trainable_state, restore_trainable_state
from utils import write_jsonl, SummaryReport

def setup_mpi():
    ''' Function to create the world communicator between nodes and also the shared memory
        communicator for intra-node communication
    '''

    ## World communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
 
    ## Communicator for the local group(shared memory domain)
    ## used to pick which processor each process that node should use.
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()
 
    return comm, rank, world_size, local_rank, local_size

def pick_device(local_rank, local_size, use_cuda):
    ''' Function to pick the appropriate device for each process based on its local rank and
        the number of available GPUs.
    '''

    if torch.cuda.is_available() and use_cuda:
        n_gpus = torch.cuda.device_count()
        if local_rank < n_gpus:
            backend = "nccl"
            torch.cuda.set_device(local_rank)
            return backend, torch.device(f"cuda:{local_rank}")
    else:
        n_gpus = 0
        backend = "gloo"
        n_cpu_ranks_on_node = max(1, local_size - n_gpus)
        cores_per_rank = max(1, len(os.sched_getaffinity(0)) // n_cpu_ranks_on_node)
        torch.set_num_threads(cores_per_rank)

        return backend, torch.device("cpu")

    n_cpu_ranks_on_node = max(1, local_size - n_gpus)
    cores_per_rank = max(1, (os.cpu_count() or 1) // n_cpu_ranks_on_node)
    torch.set_num_threads(cores_per_rank)

    return torch.device("cpu")

def contiguous_fold_chunks(n_folds, world_size):
    """ Split n_folds into world_size contiguous chunks, as equal as
        possible, with any remainder going to the LAST ran.
    """

    base = n_folds // world_size
    remainder = n_folds % world_size
    chunks = []
    start = 0
    for r in range(world_size):
        size = base if r < world_size - 1 else base + remainder
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks

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


def build_model(config, device, llm_device="cpu", train_adapters=False):
    from models.lynx import LynxBase

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = LynxBase(
        config=config,
        freeze_vit=config["freeze_vit"],
        freeze_llm=config["freeze_llm"],
        load_bridge=False,
    )

    if device.type == "cuda":
        model.vision_encoder = model.vision_encoder.to(device).half()
        model.bridge = model.bridge.to(device).half()
    else:
        model.vision_encoder = model.vision_encoder.to(device)
        model.bridge = model.bridge.to(device)

    if llm_device != "cpu":
        print("### For cross-validation, keeping LLM on CPU to match generate.py placement", flush=True)
    
    llm_device = torch.device("cpu")
    model.LLM = model.LLM.to(llm_device)

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

    return model


def train_one_fold(model, train_loader, device, epochs, trainable_params, optimizer, grad_accum_steps, grad_clip):
    model.train()
    model.vision_encoder.eval()

    history = []
    last_loss = 0.0
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
        last_loss = epoch_loss
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "global_step": global_step})
        print(f"### Finishing epoch {epoch} and aplying gradients from Step {step + 1}/{len(train_loader)}", flush=True)

    return last_loss, history


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
    comm, rank, world_size, _ , _ = setup_mpi()    
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    cudnn.benchmark = True

    seed = args.seed

    data_files = resolve_data_files(args, config)
    config_for_data = copy.deepcopy(config)
    config_for_data["test_files"] = data_files

    print("### Creating dataset for cross-validation", flush=True)
    full_dataset = create_train_dataset("eval", config_for_data)
    n = len(full_dataset)
    assert args.k_folds >= 2, "k_folds must be >= 2"
    assert args.k_folds <= n, f"k_folds must be <= dataset size (n={n})"

    if rank == 0:
        try:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"### Number of CPU threads: {torch.get_num_threads()}", flush=True)

            folds = make_kfold_indices(
                full_dataset.data,
                k=args.k_folds,
                seed=args.fold_seed,
                stratified=(not args.disable_stratified)
            )
            print("### Fold sizes:", [len(f) for f in folds], flush=True)

            error = None
        except Exception as e:
            folds, device, error = None, None, str(e)
    else:
        folds, device, error = None, None, None
    
    folds, device, error = comm.bcast((folds, device, error), root=0)

    all_predictions = []
    fold_reports = []
    num_workers = max(2, min(8, (os.cpu_count() or 1) // 2))
    summary_report = SummaryReport(
        data_files,
        seed,
        n,
        cross_val=True,
        mpi=True,
        folds=args.k_folds,
        world_size=world_size,
        num_threads=torch.get_num_threads()
    )

    fold_indices = contiguous_fold_chunks(len(folds), world_size)[rank]

    if rank == 0:
        print(f"### Using {num_workers} workers for DataLoader", flush=True)
        print(f"### Start {args.k_folds}-fold cross-validation with retraining", flush=True)
        
        global_start_time = time.time()
    rank_start_time = time.time()

    model_build_time = time.time()
    model = build_model(
            config=config,
            device=device,
            llm_device=args.llm_device,
            train_adapters=args.train_adapters
        )
    model_build_time = time.time() - model_build_time

    trainable_params, n_trainable = get_trainable_params(model)
    if n_trainable == 0:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")

    pristine_state = snapshot_trainable_state(model)

    
    for fold_idx in fold_indices:
        val_indices = folds[fold_idx]
        fold_start = time.time()
        train_indices = sorted(list(set(range(n)) - set(val_indices)))

        print(
            f"### Fold {fold_idx+1}/{args.k_folds}: train={len(train_indices)} val={len(val_indices)}",
            flush=True,
        )

        restore_trainable_state(model, pristine_state)

        train_loader_dataset_build_time = time.time()
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
        train_loader_dataset_build_time = time.time() - train_loader_dataset_build_time

        del train_dataset

        optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=config["weight_decay"])

        t0 = time.time()
        train_history = train_one_fold(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            trainable_params=trainable_params,
            optimizer=optimizer,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
        )
        train_time = time.time() - t0

        del train_loader

        t1 = time.time()
        preds, metrics = evaluate_one_fold(
            model=model,
            base_dataset=full_dataset,
            val_indices=val_indices,
            device=device,
            config=config,
            num_workers=args.num_workers,
        )
        evaluate_time = timte.time() - t1

        del full_dataset

        for p in preds:
            p["fold"] = fold_idx+1
        all_predictions.extend(preds)

        fold_report = {
            "fold": fold_idx+1,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "train_time_sec": train_time,
            "train_history": train_history,
            "metrics": metrics,
        }
        fold_reports.append(fold_report)

        fold_line = summary_report.fold(
            metrics,
            last_train_loss,
            len(train_indices),
            len(val_indices),
            fold_elapsed,
            fold_idx,
        )
        print(f"### {fold_line}", flush=True)

        if args.save_fold_checkpoints:
            save_fold_checkpoint(False, model, rank, fold_id, "X", args.checkpoint_dir)
            print(f"### Fold model saved to: {args.checkpoint_dir}", flush=True)

        del model
        torch.cuda.empty_cache()
    
    io_time = model_build_time + train_loader_dataset_build_time
    process_time = train_time + evaluate_time
    rank_time = time.time() - rank_start_time

    gathered_io_time = comm.gather(io_time, root=0)
    gathered_process_time = comm.gather(process_time, root=0)
    gathered_reports = comm.gather(fold_reports, root=0)
    gathered_predictions = comm.gather(all_predictions, root=0)
    gathered_rank_times = comm.gather(rank_time, root=0)

    if rank == 0:
        taken_time = time.time() - global_start_time
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_out = os.path.join(args.output_dir, f"cross_val_predictions_{timestamp}.jsonl")
        report_out = os.path.join(args.output_dir, f"cross_val_report_{timestamp}.json")

        all_fold_predictions = [p for ranks_preds in gathered_predictions for p in ranks_preds]
        all_fold_io_time = [p for ranks_preds in gathered_predictions for p in ranks_preds]

        if args.save_fold_predictions:
            write_jsonl(all_fold_predictions, pred_out, 'w')
            print(f"### Saved predictions: {pred_out}", flush=True)
        
        all_fold_reports = sorted(
            (fr for ranks_reports in gathered_reports for fr in ranks_reports),
            key=lambda fr: fr["fold"],
        )

        aggregate = summary_report.aggregate(all_fold_reports)
        full_report = summary_report.full(report_out, aggregate, taken_time, io_time, process_time, gathered_rank_times)
        short_report = summary_report.short(time_taken, aggregate)

        with open(report_out, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        print("### Cross-validation finished", flush=True)
        print(f"### Predictions saved to: {pred_out}", flush=True)
        print(f"### Report saved to: {report_out}", flush=True)
        print(f"### {short_report}")

if __name__ == "__main__":
    import argparse

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
    parser.add_argument("--save_fold_predictions", action="store_true")
    parser.add_argument("--save_fold_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./cv_checkpoints")

    parser.add_argument("--train_adapters", action="store_true")
    
    parser.add_argument("--disable_stratified", action="store_true")
    
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