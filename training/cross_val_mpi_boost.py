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
from torch.utils.data import Dataset, Subset

from calculate_metrics import extract_class, process_metrics
from dataset import create_train_dataset, create_loader
from dataset.train_datasets import LynxTrainDataset
from training.utils import build_model_mmap, save_fold_checkpoint, resolve_data_files
from utils import write_jsonl, summary_line

def setup_mpi():
    ''' Function to create the world communicator between nodes and also the shared memory
        communicator for intra-node communication
    '''

    # World communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
 
    # Communicator for the local group(shared memory domain)
    # used to pick which processor each process that node should use.
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
        #torch.set_device(local_rank)
        #vision_device_name = "cpu:"
        #llm_device_name = "cpu:"
        n_cpu_ranks_on_node = max(1, local_size - n_gpus)
        cores_per_rank = max(1, (os.cpu_count() or 1) // n_cpu_ranks_on_node)
        torch.set_num_threads(cores_per_rank)

        return backend, torch.device("cpu")

    #vision_device = torch.device(f"{vision_device_name}{local_rank}")
    #llm_device = torch.device(f"{llm_device_name}{local_rank}")
    
    n_cpu_ranks_on_node = max(1, local_size - n_gpus)
    cores_per_rank = max(1, (os.cpu_count() or 1) // n_cpu_ranks_on_node)
    torch.set_num_threads(cores_per_rank)

    return torch.device("cpu")

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
    rank=0
):
    model.train()
    running_loss = 0.0
    steps = len(train_loader)

    vision_dtype = next(model.vision_encoder.parameters()).dtype
    print(f"### Starting epoch", flush=True)
 
    for step, (_, vision_input, input_ids, input_atts, labels) in enumerate(train_loader, start=1):
        vision_input = vision_input.to(vision_device, dtype=vision_dtype, non_blocking=True)
 
        loss = forward_lm(model, vision_input, input_ids, input_atts, labels, llm_device)
 
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
 
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip
            )
 
        optimizer.step()
        running_loss += loss.item()
 
        if step % 30 == 0 or step == steps:
            avg = running_loss / step
            print(
                f"### Train epoch {epoch}/{epochs}, step {step}/{steps}, avg_loss={avg:.4f}",
                flush=True,
            )

    return running_loss / max(1, steps)

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

print("MODULE LOADED", flush=True)

def main(args, config):
    print("### Building dist. architecture", flush=True)
    comm, rank, world_size, local_rank, local_size = setup_mpi()
    _, device = pick_device(local_rank, local_size, False)
    vision_device = llm_device = device
    #print(f"### LLM device: {device}")

    print("### Loading dataset for cross-validation", flush=True)
    full_dataset = create_train_dataset("eval", config)
    dataset_size = len(full_dataset)

    if rank == 0:
        try:
            print("### Starting cross-validation with MPI", flush=True)
            print(f"### args: {args}", flush=True)
            print(f"### config: {config}", flush=True)
        
            print("### Preparing seed main", flush=True)
            seed = args.seed + rank
            torch.manual_seed(seed)
            np.random.seed(seed)
            cudnn.benchmark = True

            print("### Creating folds for cross-validation", flush=True)
            assert args.k_folds >= 2, "k_folds must be >= 2"
            assert args.k_folds < dataset_size, f"k_folds must be <= dataset size (n={dataset_size})"            

            folds = make_stratified_k_folds(full_dataset.data, args.k_folds, seed)
            print("### Fold sizes:", [len(f) for f in folds], flush=True)

            error = None
        except Exception as e:
            folds, error = None, str(e)
    else:
        folds, error = None, None
    
    folds, error = comm.bcast((folds, error), root=0)
    if error is not None:
        raise ValueError(f"Error in rank 0 during setup: {error}")

    print(f"### rank {rank}: device={device}", flush=True)
    fold_indices = contiguous_fold_chunks(len(folds), world_size)[rank]
    rank_predictions = []
    fold_reports = []
    using_gpu = device.type == "cuda"

    if rank == 0:
        global_start = time.time()
    
    model = build_model_mmap(config, vision_device, llm_device)
    trainable_params, n_trainable = get_trainable_params(model)
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Set --unfreeze_vit and/or --unfreeze_llm if needed.")
    pristine_state = snapshot_trainable_state(model)
    
    for fold_idx in fold_indices: #in enumerate(folds, start=1):
        val_indices = folds[fold_idx]
        fold_start = time.time()

        train_set = set(range(dataset_size)) - set(val_indices)
        train_indices = sorted(list(train_set))

        restore_trainable_state(model, pristine_state)

        start_dataset = time.time()
        train_dataset = LynxTrainDataset(
            full_dataset,
            train_indices,
            max_train_tokens=args.max_train_tokens,
        )
        val_dataset = Subset(full_dataset, val_indices)

        time_dataset = time.time() - start_dataset

        start_loader = time.time()
        train_loader, val_loader  = create_loader(
            [train_dataset, val_dataset],
            batch_size=[args.batch_size_train, args.batch_size_train],
            num_workers=[args.num_workers, args.num_workers],
            collate_fns=[train_dataset.collate_fn, full_dataset.collate_fn],
            shuffles=[True,True],
            pin_memories=[using_gpu, using_gpu]
            )
        
        time_loader = time.time() - start_loader
        
        # Removing unused datasets to free memory
        del train_dataset
        del val_dataset

        optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"], weight_decay=config["weight_decay"])

        best_loss = config.get("best_loss", float("inf"))
        last_train_loss = 0.0

        # Run all the epochs for each fold
        for epoch in range(1, args.epochs + 1):
            last_train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                vision_device,
                llm_device,
                epoch,
                args.epochs,
                grad_clip=1.0
            )

            if last_train_loss < best_loss:
                best_loss = last_train_loss
                if args.save_fold_checkpoints:
                    save_fold_checkpoint(True, model, local_rank, fold_idx, epoch, args.checkpoint_dir, optimizer)

        predictions = evaluate_generation(model, val_loader, vision_device, llm_device, config)
        val_annotations = [full_dataset.data[i] for i in val_indices]
        metrics = process_metrics(val_annotations, predictions)

        # Removing optimizer and loaders to free memory before next fold
        del optimizer
        del train_loader
        del val_loader

        for pred in predictions:
            pred["fold"] = fold_idx

        rank_predictions.extend(predictions)

        fold_elapsed = time.time() - fold_start
        fold_line = summary_line(
            metrics,
            last_train_loss,
            len(train_indices),
            len(val_indices),
            fold_elapsed,
            fold_idx,
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

        torch.cuda.empty_cache()

    del model

    gathered_reports = comm.gather(fold_reports, root=0)
    gathered_predictions = comm.gather(rank_predictions, root=0)

    if rank == 0:
        total_time = time.time() - global_start

        all_fold_reports = sorted(
            (fr for ranks_reports in gathered_reports for fr in ranks_reports),
            key=lambda fr: fr["fold"],
        )
        all_predictions = [p for ranks_preds in gathered_predictions for p in ranks_preds]

        metric_names = ["accuracy", "f1_fire", "f1_nofire", "f1_macro"]
        aggregate = {}
        for name in metric_names:
            values = [fr["metrics"][name] for fr in all_fold_reports]
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values))

        if args.save_fold_predictions:
            write_jsonl(all_predictions, args.output_predictions)
            print(f"### Saved predictions: {args.output_predictions}", flush=True)

        final_report = {
            "k_folds": args.k_folds,
            "dataset_size": dataset_size,
            "data_files": data_files,
            "epochs": args.epochs,
            "batch_size_train": args.batch_size_train,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "fold_reports": all_fold_reports,
            "aggregate": aggregate,
            "total_time": str(dt.timedelta(seconds=int(total_time))),
        }

        with open(args.report_path, "w") as f:
            json.dump(final_report, f, indent=2)

        print(f"### Saved cross-validation report: {args.report_path}", flush=True)
        print("### Aggregate:", json.dumps(aggregate, indent=2), flush=True)
        print(f"### Total time: {str(dt.timedelta(seconds=int(total_time)))}", flush=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_adapters", action="store_true")
    
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm_device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--data_files", type=str, nargs="*", default=None)
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
    parser.add_argument("--report_path", type=str, default="cross_val_report.json")
    parser.add_argument("--save_fold_predictions", action="store_true")
    parser.add_argument("--save_fold_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./cv_checkpoints")

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

    main(args, config)