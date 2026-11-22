import copy
import json

import os
import numpy as np

import time
import datetime as dt
from datetime import datetime

from mpi4py import MPI
from ruamel.yaml import YAML

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, Subset

from calculate_metrics import extract_class, process_metrics
from dataset import create_train_dataset, create_loader
from dataset.train_datasets import TrainFoldDataset
from training.utils import resolve_data_files, snapshot_trainable_state, restore_trainable_state
from utils import write_jsonl, SummaryReport

TAG_REQ, TAG_WORK, TAG_STOP = 1, 2, 3

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

def setup_mpi():
    ''' Function to create the world communicator between nodes and also the shared memory
        communicator for intra-node communication
    '''

    ## World communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    if world_size < 2:
        raise RuntimeError(f"### Need at least 2 ranks, found {world_size}")
    
    ## Communicator for the local group(shared memory domain)
    ## used to pick which processor each process that node should use.
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()

    dispatcher = local_comm.allreduce(1 if rank == 0 else 0, op=MPI.SUM)
    worker_local_idx = local_rank - dispatcher
 
    return comm, rank, world_size, worker_local_idx

def pick_worker_devices(worker_local_idx):
    """Vision encoder (+ bridge) on a GPU if one is free for this worker,
    LLM always on CPU -- matching your current split."""
    if torch.cuda.is_available() and worker_local_idx < torch.cuda.device_count():
        torch.cuda.set_device(worker_local_idx)
        vision_device = torch.device(f"cuda:{worker_local_idx}")
    else:
        vision_device = torch.device("cpu")
    llm_device = torch.device("cpu")
 
    # Every worker runs the LLM on CPU, so every worker caps its threads.
    # Prefer OMP_NUM_THREADS from mpirun (-x), else the cores this process
    # is actually allowed to use (respects --bind-to), not os.cpu_count().
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    n_threads = int(os.environ.get("OMP_NUM_THREADS", 0)) or physical or len(os.sched_getaffinity(0))
    torch.set_num_threads(max(1, n_threads))
    return vision_device, llm_device

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

def write_final_report(save_fold_predictions, pred_out, report_out, fold_reports, all_predictions):

    if save_fold_predictions:
        write_jsonl(all_predictions, pred_out, 'w')
        print(f"### Saved predictions: {pred_out}", flush=True)

    aggregate = summary_report.aggregate(fold_reports)
    full_report = summary_report.full(report_out, aggregate, taken_time, gathered_rank_times)
    short_report = summary_report.short(time_taken, aggregate)

    write_json(full_report, report_out)
    write_txt_doc(short, "stats.txt")

    print("### Cross-validation finished", flush=True)
    print(f"### Predictions saved to: {pred_out}", flush=True)
    print(f"### Report saved to: {report_out}", flush=True)
    print(f"### {short_report}")

def train_one_epoch(model, optimizer, train_loader, device, epoch, epochs, grad_accum_steps, grad_clip):
    model.train()
    running_loss = 0.0
    steps = len(train_loader)

    vision_dtype = next(model.vision_encoder.parameters()).dtype

    for step, batch in enumerate(train_loader):
        loss = forward_train_loss(model, batch, device)
        loss = loss / grad_accum_steps
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        ## If we accumulated grad_accum_steps, apply gradients
        if (step + 1) % grad_accum_steps == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            running_loss += loss.item() * grad_accum_steps
        
        if step % 30 == 0 or step == steps:
            avg = running_loss / step
            print(f"### Train epoch {epoch}/{epochs}, step {step}/{steps}, avg_loss={avg:.4f}", flush=True)
    
    if n_steps % grad_accum_steps != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
    
    return running_loss / max(1, steps)

def train_one_fold(model, train_loader, device, epochs, trainable_params, lr, weight_decay, grad_accum_steps=1, grad_clip=1.0):
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    ## Loss metrics
    best_loss = 0.0
    last_loss = 0.0

    start_training = time.time()

    ## Run all the epochs for this fold
    for epoch in range(1, config["train_epochs"] + 1):
        print(f"### Starting epoch", flush=True)
        last_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            vision_device,
            llm_device,
            epoch,
            epochs,
            grad_clip=1.0
        )
        if last_loss < best_loss:
            best_loss = last_loss
    end_training = time.time() - start_training

    return last_loss, best_loss, end_training

def run_one_fold(config, device, absl_best_loss, model, full_dataset, fold_idx, epochs, folds, rank):
    ## Generating indices for training/validation split
    val_indices = folds[fold_idx]
    train_indices = sorted(list(set(range(len(full_dataset))) - set(val_indices)))

    using_gpu = device.type == "cuda"

    train_dataset_loader_time = time.time()
    train_dataset = TrainFoldDataset(
        base_dataset=full_dataset,
        indices=train_indices,
        max_total_tokens=args.max_train_tokens,
    )
    train_loader  = create_loader(
        [train_dataset],
        batch_size=[config["batch_size_train"]],
        num_workers=[config["num_workers"]],
        collate_fns=[train_dataset.collate_fn],
        shuffles=[True],
        pin_memories=[using_gpu]
    )
    train_dataset_loader_time = time.time() - train_dataset_loader_time

    del train_dataset

    from training.utils import get_trainable_params
    trainable_params, _ = get_trainable_params(model)
    last_loss, best_loss, training_time = train_one_fold(
        model,
        train_loader,
        device,
        absl_best_loss,
        epochs,
        trainable_params, 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        grad_accum_steps=config["grad_accum_steps"]
    )
    
    ## If it has the best loss, save this model
    if best_loss < absl_best_loss:
        if args.save_fold_checkpoints:
            save_fold_checkpoint(False, model, rank, fold_idx, epoch, args.checkpoint_dir)

    preds, metrics = evaluate_one_fold(
        model=model,
        base_dataset=full_dataset,
        val_indices=val_indices,
        device=device,
        config=config,
        num_workers=args.num_workers,
    )

    for p in preds:
        p["fold"] = fold_idx + 1

    fold_report_line = summary_report.fold_line(
        (fold_idx + 1),
        len(train_indices),
        len(val_indices),
        train_time,
        last_loss,
        metrics
    )
    fold_reports.append(fold_report_line)
    print(fold_report_line, flush=True)

    return {"best_loss": best_loss, "report": fold_report_line, "predictions":preds}

def main(args, config):
    comm, rank, world_size, worker_idx = setup_mpi()    
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    cudnn.benchmark = True

    ## Extracting data files in case of conflict
    data_files = resolve_data_files(args, config)
    config_for_data = copy.deepcopy(config)
    config_for_data["test_files"] = data_files

    print("### Creating dataset for cross-validation", flush=True)
    full_dataset = create_train_dataset("eval", config_for_data)
    dataset_size = len(full_dataset)

    folds, absl_best_loss, error = None, None, None
    if rank == 0:
        try:
            print("### Starting cross-validation with MPI", flush=True)
            print(f"### args: {args}", flush=True)
            print(f"### config: {config}", flush=True)

            absl_best_loss = config.get("best_loss", float("inf"))
        
            print("### Preparing seed main", flush=True)
            seed = args.seed + rank
            torch.manual_seed(seed)
            np.random.seed(seed)
            cudnn.benchmark = True

            ## Checking if k-fold are feasible
            print("### Creating folds for cross-validation", flush=True)
            assert args.k_folds >= 2, "k_folds must be >= 2"
            assert args.k_folds < dataset_size, f"k_folds must be <= dataset size (n={dataset_size})"            

            folds = make_stratified_k_folds(full_dataset.data, args.k_folds, seed)
            print("### Fold sizes:", [len(f) for f in folds], flush=True)

            error = None
        except Exception as e:
            folds, absl_best_loss, error = None, None, str(e)
    
    folds, absl_best_loss, error = comm.bcast((folds, absl_best_loss, error), root=0)
    if error is not None:
        raise ValueError(f"Error in rank 0 during setup: {error}")

    ## Spawn workers for training
    if rank == 0:
        pending = list(range(len(folds)))
        fold_reports, all_predictions = [], []
        active = world_size - 1

        global_start_time = time.time()
        while active:
            st = MPI.Status()
            while not comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQ, status=st):
                time.sleep(0.2)                      # idle without burning a core
            src = st.Get_source()

            msg = comm.recv(source=src, tag=TAG_REQ)
            if isinstance(msg, dict) and "error" in msg:
                print(f"### worker {src} failed: {msg['error']}", flush=True)
                comm.Abort(1)

            if msg is not None:
                if not (isinstance(msg, dict) and "report" in msg and "predictions" in msg):
                    keys = list(msg) if isinstance(msg, dict) else None
                    print(f"### dispatcher: unexpected message from rank {src}: "
                        f"type={type(msg).__name__}, keys={keys}", flush=True)
                    comm.Abort(1)
                fold_reports.append(msg["report"])
                all_predictions.extend(msg["predictions"])
                if msg["best_loss"] < absl_best_loss:
                    absl_best_loss = msg["best_loss"]
            if pending:
                comm.send((absl_best_loss, pending.pop(0)), dest=st.Get_source(), tag=TAG_WORK)
            else:
                comm.send(None, dest=st.Get_source(), tag=TAG_STOP)
                active -= 1
        
        taken_time = time.time() - global_start_time
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_out = os.path.join(args.output_dir, f"cross_val_predictions_{timestamp}.jsonl")
        report_out = os.path.join(args.output_dir, f"cross_val_report_{timestamp}.json")

        write_final_report(args.save_fold_predictions, pred_out, report_out, fold_reports, all_predictions)

    ## Worker code
    else:
        vision_device, llm_device = pick_worker_devices(worker_idx)
        model = build_model(config, vision_device, llm_device)
        snapshot_trainable_state(model)

        result = None
        while True:
            comm.send(result, dest=0, tag=TAG_REQ)
            result = None
            restore_trainable_state(model, pristine_state)

            st = MPI.Status()
            absl_best_loss, fold_idx = comm.recv(source=0, tag=MPI.ANY_TAG, status=st)
            if st.Get_tag() == TAG_STOP:
                break
            # If we continue:
            try:
                result = run_one_fold(config, vision_device, absl_best_loss, model, full_dataset, fold_idx, args.epochs, folds, rank)
                if result["best_loss"] < absl_best_loss:
                    absl_best_loss = result["best_loss"]
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                comm.send({"error":f"fold{fold_idx+1}:{e!r}"}, dest=0, tag=TAG_REQ)
                raise
            
        del model


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
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_tokens", type=int, default=256) # 128
    parser.add_argument("--output_dir", type=str, default="./cv_outputs")

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