import copy
import json

import os
import random
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
from dataset.train_datasets import TrainFoldDataset, CachedTrainDataset, CachedEvalDataset
from training.utils import resolve_data_files, make_optimizer, VisionCache, vision_cache_key, build_vision_cache, bridge_from_cache, setup_node_vision_cache, process_use_cache # snapshot_trainable_state, restore_trainable_state
from utils import write_jsonl, SummaryReport

TAG_REQ, TAG_WORK, TAG_STOP, TAG_BUILT, TAG_GO = 1, 2, 3, 4, 5

def report_built(comm, build_seconds, error=None):
    """ WORKER side: tell rank 0 this worker finished (or failed) building,
        then block until rank 0 says every worker is done.
    """
    comm.send(
        {"rank": comm.Get_rank(), "host": MPI.Get_processor_name(),
         "seconds": build_seconds, "error": error},
        dest=0, tag=TAG_BUILT,
    )
    go = comm.recv(source=0, tag=TAG_GO)
    if not go:
        raise RuntimeError("Another worker failed to build its model; stopping.")

def wait_until_all_built(comm):
    """ DISPATCHER (rank 0) side: collect a 'built' message from every worker,
        print who took how long, then release all workers at once.
    """
    n_workers = comm.Get_size() - 1
    reports, start = [], time.time()
    while len(reports) < n_workers:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_BUILT)
        reports.append(msg)
        status = "FAILED: " + msg["error"] if msg["error"] else f"{msg['seconds']:.0f}s"
        print(f"### build done {len(reports)}/{n_workers}: rank {msg['rank']} "
              f"on {msg['host']} ({status})", flush=True)
 
    all_ok = all(r["error"] is None for r in reports)
    for r in range(1, comm.Get_size()):
        comm.send(all_ok, dest=r, tag=TAG_GO)
    print(f"### all models built; last one finished {time.time() - start:.0f}s "
          f"after the first report", flush=True)
    if not all_ok:
        raise RuntimeError("At least one worker failed to build its model.")
    
    seconds_list = [msg["seconds"] for msg in reports]
    ranks_list = [msg["rank"] for msg in reports]
    return seconds_list, ranks_list

def forward_train_loss(model, batch, device):
    _, vision_input, input_ids, input_atts, labels = batch

    llm_device = next(model.LLM.parameters()).device
    input_ids = input_ids.to(llm_device)
    input_atts = input_atts.to(llm_device)
    labels = labels.to(llm_device)

    text_embeds = model.embed_tokens(input_ids)
    vision_dtype = next(model.vision_encoder.parameters()).dtype

    if vision_input is not None:
        vision_input = vision_input.to(
            next(model.vision_encoder.parameters()).device,
            dtype=vision_dtype,
            non_blocking=True
        )
        with torch.set_grad_enabled(not model.freeze_vit):
            vision_embeds, vision_atts = model.get_vision_embeds(vision_input)

        bridge_dtype = next(model.bridge.parameters()).dtype
        v2t_feats, v2t_atts = model.bridge(vision_embeds=vision_embeds.to(bridge_dtype), vision_atts=vision_atts)
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

def forward_train_loss_cached(model, batch, vision_cache):
    _, vision_embeds, input_ids, input_atts, labels = batch

    llm_device = next(model.LLM.parameters()).device
    input_ids = input_ids.to(llm_device)
    input_atts = input_atts.to(llm_device)
    labels = labels.to(llm_device)

    text_embeds = model.embed_tokens(input_ids)
    v2t_feats, v2t_atts = bridge_from_cache(model, vision_embeds, vision_cache,
                                             llm_device, text_embeds.dtype)

    ignore = torch.full((labels.shape[0], v2t_feats.shape[1]), -100,
                        dtype=labels.dtype, device=llm_device)
    outputs = model.LLM(
        inputs_embeds=torch.cat([v2t_feats, text_embeds], dim=1),
        attention_mask=torch.cat([v2t_atts, input_atts], dim=1),
        labels=torch.cat([ignore, labels], dim=1),
        return_dict=True,
    )
    return outputs.loss

def reinit_adapters(model):
    """ Re-initialize every Lynx output_adapter IN PLACE.
        Harmless if the adapters are frozen (they just get reset to a fresh init too.
        If you want frozen adapters to keep their PRETRAINED values instead, skip this
        call when they aren't trainable
    """
    n = 0
    for m in model.LLM.modules():
        if type(m).__name__ == "Adapter":
            m.adapter_norm_before.reset_parameters()
            m.adapter_down.apply(m._init_weights)
            m.adapter_up.apply(m._init_weights)
            with torch.no_grad():
                m.scaling.fill_(1.0)
            n += 1
    return n

def reinit_bridge(model, config, vision_device):
    """ Rebuild the bridge from scratch through LynxBase's own method, the
        same call its __init__ makes. Produces NEW Parameter objects, so the
        optimizer must be (re)created after this -- run_fold does that.
    """
    old_requires_grad = {n: p.requires_grad for n, p in model.bridge.named_parameters()}
    bridge, _missing = model.build_bridge(config, load_params=False)
    print({str(p.dtype) for n, p in model.named_parameters() if p.requires_grad})
    print(next(model.bridge.parameters()).dtype)
    model.bridge = bridge.to(vision_device)
    return old_requires_grad

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
 
    return comm, rank, world_size, worker_local_idx, local_comm

def pick_worker_devices(worker_local_idx):
    """ Vision encoder (+ bridge) on a GPU if one is free for this worker,
        LLM always on CPU -- matching your current split.
    """
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

def write_final_report(save_fold_predictions, summary_report, pred_out, report_out, fold_reports, all_predictions, taken_time):

    if save_fold_predictions:
        write_jsonl(all_predictions, pred_out, 'w')
        print(f"### Saved predictions: {pred_out}", flush=True)

    aggregate = summary_report.aggregate(fold_reports)
    full_report = summary_report.full(report_out, fold_reports, taken_time)
    short_report = summary_report.short(taken_time, aggregate)

    write_json(full_report, report_out)
    write_txt_doc(short, "stats.txt")

    print("### Cross-validation finished", flush=True)
    print(f"### Predictions saved to: {pred_out}", flush=True)
    print(f"### Report saved to: {report_out}", flush=True)
    print(f"### {short_report}")

@torch.no_grad()
def evaluate_one_fold(model, base_dataset, val_indices, device, config):
    val_subset = Subset(base_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size_train,
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=False,
        collate_fn=base_dataset.collate_fn,
        drop_last=False,
    )

    preds = []

    model.eval()
    with torch.inference_mode():
        for idx, vision_input, input_ids, input_atts in val_loader:

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

@torch.no_grad()
def evaluate_generation_cached(model, base_dataset, config, val_indices, vision_cache):
    val_subset = CachedEvalDataset(base_dataset, val_indices, vision_cache)
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size_train,
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=False,
        collate_fn=base_dataset.collate_fn,
        drop_last=False,
    )

    llm_device = next(model.LLM.parameters()).device
    preds = []

    model.eval()
    with torch.inference_mode():
        for idx, vision_embeds, input_ids, input_atts in val_loader:
            input_ids = input_ids[:, -128:].to(llm_device)
            input_atts = input_atts[:, -128:].to(llm_device)

            text_embeds = model.embed_tokens(input_ids)
            v2t_feats, v2t_atts = bridge_from_cache(model, vision_embeds, vision_cache, 
                                                    llm_device, text_embeds.dtype)

            outputs = model.LLM.generate(
                inputs_embeds=torch.cat([v2t_feats, text_embeds], dim=1),
                attention_mask=torch.cat([v2t_atts, input_atts], dim=1),
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
            for i, text in zip(idx, model.tokenizer.batch_decode(outputs, skip_special_tokens=True)):
                preds.append({"index": i, "text_output": text.strip()})

    annotations = [base_dataset.data[i] for i in val_indices]
    metrics = process_metrics(annotations, preds)
    return preds, metrics

def train_one_epoch(model, optimizer, train_loader, device, epoch, epochs, grad_accum_steps, grad_clip, vision_cache):
    import math

    running_loss = 0.0
    steps = len(train_loader)
    n_steps, n_micro = 0, 0

    if vision_cache == None:
        vision_dtype = next(model.vision_encoder.parameters()).dtype

    from training.utils import get_trainable_params
    trainable_params, _ = get_trainable_params(model)

    model.train()
    for step, batch in enumerate(train_loader):
        if vision_cache == None:
            loss = forward_train_loss(model, batch, device) / grad_accum_steps
        else:
            loss = forward_train_loss_cached(model, batch, vision_cache) / grad_accum_steps

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        ## Calculate true loss value
        loss_val = loss.item() * grad_accum_steps

        ## Assert loss value is not infinite
        if not math.isfinite(loss_val):
            labels = batch[-1]
            raise RuntimeError(
                f"non-finite loss {loss_val} at step {step}; "
                f"supervised tokens in batch = {(labels != -100).sum().item()}, "
                f"LLM dtype = {next(model.LLM.parameters()).dtype}"
            )

        running_loss += loss_val
        n_micro += 1

        ## If we accumulated grad_accum_steps or is the last step, apply gradients
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps:
            if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()

            bad = [n for n, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
            if bad:
                raise RuntimeError(f"weights became non-finite after optimizer.step(): {bad[:5]}")

            n_steps += 1
        
        if (step % 30 == 0 or (step + 1) == steps) and step != 0:
            avg = running_loss / step
            print(f"### Train epoch {epoch}/{epochs}, step {step}/{steps}, avg_loss={running_loss:.4f}", flush=True)
    
    if n_steps % grad_accum_steps != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
    
    return running_loss / max(1, steps)

def train_one_fold(model, train_loader, device, epochs, trainable_params, lr, weight_decay, grad_accum_steps=1, grad_clip=1.0, vision_cache=None):
    optimizer = make_optimizer(trainable_params, lr, weight_decay, True)
    
    ## Loss metrics
    best_loss = 0.0
    last_loss = 0.0

    start_training = time.time()

    ## Run all the epochs for this fold
    for epoch in range(1, config["train_epochs"]):
        print(f"### Starting epoch", flush=True)
        last_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            epochs,
            grad_accum_steps,
            grad_clip,
            vision_cache
        )
        if last_loss < best_loss:
            best_loss = last_loss
    end_training = time.time() - start_training

    return last_loss, best_loss, end_training

def run_one_fold(use_cache, summary_report, config, device, absl_best_loss, model, full_dataset, fold_idx, epochs, folds, rank, vision_cache=None):
    ## Generating indices for training/validation split
    val_indices = folds[fold_idx]
    train_indices = sorted(list(set(range(len(full_dataset))) - set(val_indices)))

    using_gpu = device.type == "cuda"
    
    if not use_cache or vision_cache==None: # create flag for caching vision model
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
        )[0]
        train_dataset_loader_time = time.time() - train_dataset_loader_time

        del train_dataset

        old_rg = reinit_bridge(model, config, device)
        new_rg = {n: p.requires_grad for n, p in model.bridge.named_parameters()}
        assert new_rg == old_rg, f"{tag} rebuilt bridge has different requires_grad than the original"
        # if adapters_trainable
        reinit_adapters(model)
        
        from training.utils import get_trainable_params
        trainable_params, _ = get_trainable_params(model)

        train_time = time.time()
        last_loss, best_loss, training_time = train_one_fold(
            model,
            train_loader,
            device,
            epochs,
            trainable_params, 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            grad_accum_steps=config["grad_accum_steps"],
            vision_cache=None,
        )
        train_time = time.time() - train_time
        
        ## If it has the best loss, save this model
        if best_loss < absl_best_loss:
            if args.save_fold_checkpoints:
                save_fold_checkpoint(False, model, rank, fold_idx, epoch, args.checkpoint_dir)
        
        preds, metrics = evaluate_one_fold(
            model=model,
            base_dataset=full_dataset,
            val_indices=val_indices,
            device=device,
            config=config
        )

    else:
        train_dataset = CachedTrainDataset(full_dataset, train_indices, vision_cache,
                                        max_train_tokens=args.max_train_tokens)
        train_loader  = create_loader(
            [train_dataset],
            batch_size=[config["batch_size_train"]],
            num_workers=[config["num_workers"]],
            collate_fns=[train_dataset.collate_fn],
            shuffles=[True],
            pin_memories=[using_gpu]
        )[0]
        
        del train_dataset
        
        from training.utils import get_trainable_params
        trainable_params, _ = get_trainable_params(model)

        train_time = time.time()
        last_loss, best_loss, training_time = train_one_fold(
            model,
            train_loader,
            device,
            epochs,
            trainable_params, 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            grad_accum_steps=config["grad_accum_steps"],
            vision_cache=vision_cache,
        )
        train_time = time.time() - train_time
        
        ## If it has the best loss, save this model
        if best_loss < absl_best_loss:
            if args.save_fold_checkpoints:
                save_fold_checkpoint(False, model, rank, fold_idx, epoch, args.checkpoint_dir)
        
        preds, metrics = evaluate_generation_cached(
            model=model,
            base_dataset=full_dataset,
            val_indices=val_indices,
            config=config,
            vision_cache=vision_cache
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
    print(fold_report_line, flush=True)

    fold_report = {
            "fold": fold_idx+1,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "train_time_sec": train_time,
            "metrics": metrics,
    }
    return {"best_loss": best_loss, "report": fold_report, "predictions":preds}

def main(args, config):
    comm, rank, world_size, worker_idx, local_comm = setup_mpi()    
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    cudnn.benchmark = True

    ## Extracting data files in case of conflict
    data_files = resolve_data_files(args, config)
    config_for_data = copy.deepcopy(config)
    config_for_data["test_files"] = data_files

    print("### Creating dataset for cross-validation", flush=True)
    full_dataset = create_train_dataset(config_for_data)
    dataset_size = len(full_dataset)

    if True: # create flag for caching vision model
        key = vision_cache_key(config, data_files)
        cache_hosts = set(args.vision_cache_hosts.split(",")) if args.vision_cache_hosts else set()

    folds, absl_best_loss, summary_report, error = None, None, None, None
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
            random.seed(seed)
            cudnn.benchmark = True

            summary_report = SummaryReport(
                data_files,
                seed,
                dataset_size,
                cross_val=True,
                mpi=True,
                folds=args.k_folds,
                world_size=world_size,
                num_threads=torch.get_num_threads()
            )

            ## Checking if k-fold are feasible
            print("### Creating folds for cross-validation", flush=True)
            assert args.k_folds >= 2, "k_folds must be >= 2"
            assert args.k_folds < dataset_size, f"k_folds must be <= dataset size (n={dataset_size})"            

            folds = make_stratified_k_folds(full_dataset.data, args.k_folds, seed)
            print("### Fold sizes:", [len(f) for f in folds], flush=True)

            error = None
        except Exception as e:
            folds, absl_best_loss, summary_report, error = None, None, None, str(e)
    
    folds, absl_best_loss, summary_report, error = comm.bcast((folds, absl_best_loss, summary_report, error), root=0)
    if error is not None:
        print('\a')
        raise ValueError(f"Error in rank 0 during setup: {error}")

    ## Spawn workers for training
    if rank == 0:
        pending = list(range(len(folds)))
        fold_reports, all_predictions = [], []
        active = world_size - 1

        if True: # create flag for caching vision model
            seconds_list, ranks_list = wait_until_all_built(comm)
            setup_node_vision_cache(None, full_dataset, local_comm, is_worker=False,
                            cache_hosts=cache_hosts, cache_dir=args.vision_cache_dir, key=key)

        global_start_time = time.time()
        while active:
            st = MPI.Status()
            while not comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_REQ, status=st):
                time.sleep(0.2)                      # idle without burning a core
            src = st.Get_source()

            msg = comm.recv(source=src, tag=TAG_REQ)
            if isinstance(msg, dict) and "error" in msg:
                print(f"### worker {src} failed: {msg['error']}", flush=True)
                print('\a')
                comm.Abort(1)

            if msg is not None:
                if not (isinstance(msg, dict) and "report" in msg and "predictions" in msg):
                    keys = list(msg) if isinstance(msg, dict) else None
                    print(f"### Unexpected message from rank {src}: type={type(msg).__name__}, keys={keys}", flush=True)
                    print('\a')
                    comm.Abort(1)
                fold_reports.append(msg["report"])
                all_predictions.extend(msg["predictions"])
                if msg["best_loss"] < absl_best_loss:
                    absl_best_loss = msg["best_loss"]
            if pending:
                comm.send((absl_best_loss, pending.pop(0)), dest=st.Get_source(), tag=TAG_WORK)
            else:
                comm.send((None, None), dest=st.Get_source(), tag=TAG_STOP)
                active -= 1
        
        taken_time = time.time() - global_start_time
        
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_out = os.path.join(args.output_dir, f"cross_val_predictions_{timestamp}.jsonl")
        report_out = os.path.join(args.output_dir, f"cross_val_report_{timestamp}.json")

        for i in range(0, len(seconds_list)):
            print(f"### Time taken to build by rank: {str(dt.timedelta(seconds=int(seconds_list[i])))} {ranks_list[i]}", flush=True)

        write_final_report(args.save_fold_predictions, summary_report, pred_out, report_out, fold_reports, all_predictions, taken_time)

    ## Worker code
    else:
        vision_device, llm_device = pick_worker_devices(worker_idx)

        host = MPI.Get_processor_name()
        use_cache = process_use_cache(host, cache_hosts, True)
        try:
            model_build_time = time.time()
            model = build_model(config, vision_device, llm_device)
            #pristine_state = snapshot_trainable_state(model)
            model_build_time = time.time() - model_build_time
            print(f"### Model built, took {model_build_time} seconds", flush=True)
            error = None
        except Exception as e:
                model, model_build_time, error = None, None, repr(e)
        
        if error == None and True: # create flag for caching vision model
            report_built(comm, model_build_time, error)
            vision_cache = setup_node_vision_cache(None, full_dataset, local_comm, is_worker=True,
                            cache_hosts=cache_hosts, cache_dir=args.vision_cache_dir, key=key)

        result = None
        while True:
            comm.send(result, dest=0, tag=TAG_REQ)
            result = None
            
            st = MPI.Status()
            absl_best_loss, fold_idx = comm.recv(source=0, tag=MPI.ANY_TAG, status=st)
            if st.Get_tag() == TAG_STOP:
                break
            # If we continue:
            try:
                result = run_one_fold(use_cache, summary_report, config, vision_device, absl_best_loss, model, full_dataset, fold_idx, args.epochs, folds, rank, vision_cache)
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
    parser.add_argument("--seed", type=int, default=12)

    parser.add_argument("--data_files", type=str, nargs="*", default=None)
    parser.add_argument("--k_folds", type=int, default=5, required=True)
    parser.add_argument("--fold_seed", type=int, default=123)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_tokens", type=int, default=256) # 128

    parser.add_argument("--output_predictions", type=str, default=None)
    parser.add_argument("--report_path", type=str, default="cross_val_report.json")
    parser.add_argument("--disable_stratified", action="store_true")
    parser.add_argument("--save_fold_predictions", action="store_true")
    parser.add_argument("--save_fold_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./cv_checkpoints")
    parser.add_argument("--vision_cache_hosts", type=str)
    parser.add_argument("--vision_cache_dir", type=str, default="./vit_cache")
    parser.add_argument("--output_dir", type=str, default="./cv_outputs")
    
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