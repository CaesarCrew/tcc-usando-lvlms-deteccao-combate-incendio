import numpy as np
import time
import datetime as dt
import torch
from torch.utils.data import Subset

from calculate_metrics import process_metrics
from utils import write_jsonl_per_fold
from dataset import create_dataset, create_loader


def make_k_folds(n, k, seed=123, shuffle=True):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    folds = np.array_split(indices, k)  # nearly equal sizes
    return folds

def make_cross_validation(test_dataset, model, device, config, k=5, fold_seed=123, save_per_fold=False):
    print("### Setting up k-fold cross-validation", flush=True)
    n = len(test_dataset)
    
    assert k <= n, f"k_folds must be <= dataset size (n={n})"

    folds = make_k_folds(n, k, fold_seed)

    start_time = time.time()
    print(f"### Start {k}-fold evaluating (outcome CV)", flush=True)

    all_predictions = []
    per_fold_paths = []
    per_fold_times = []
    prediction_test_data = ''

    from generate import evaluation  # to avoid circular import

    for fold_i, fold_indices in enumerate(folds):
        print(f"### Fold {fold_i+1}/{k}: n={len(fold_indices)}", flush=True)
        if fold_i == 0:
            prediction_test_data = 'Images per fold: ' + str(len(fold_indices))

        fold_ds = Subset(test_dataset, fold_indices.tolist())
        fold_annotations = [test_dataset.data[i] for i in fold_indices.tolist()]

        fold_loader = create_loader([fold_ds],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    collate_fns=[test_dataset.collate_fn])[0]
        
        start_time_fold = time.time()
        fold_preds = evaluation(model, fold_loader, device, config)
        fold_time = time.time() - start_time_fold
        fold_time_str = str(dt.timedelta(seconds=int(fold_time)))
        per_fold_times.append(fold_time_str)

        fold_metrics = process_metrics(fold_annotations, fold_preds)
        fold_metrics_i = "Fold {}/{} metrics: Accuracy={:.4f}, F1 fire={:.4f}, F1 nofire={:.4f}, f1_macro={:.4f}, MCC fire:{:.4f} MCC no fire:{:.4f}".format(
                fold_i + 1,
                k,
                fold_metrics["accuracy"],
                fold_metrics["f1_fire"],
                fold_metrics["f1_nofire"],
                fold_metrics["f1_macro"],
                fold_metrics["mcc_fire"],
                fold_metrics["mcc_nofire"]
            )
        print(f"###{fold_metrics_i}", flush=True)
        prediction_test_data = prediction_test_data + '\n' +  fold_metrics_i

        # tag fold id
        for p in fold_preds:
            p["fold"] = fold_i

        all_predictions.extend(fold_preds)
        if save_per_fold:
            per_fold_paths.append(write_jsonl_per_fold(fold_preds, output_path))


        # To prevent GPU memory issues, we can clear the model and empty cache after each fold
        torch.cuda.empty_cache()
    

    total_time = time.time() - start_time
    total_time_str = 'Time {}'.format(str(dt.timedelta(seconds=int(total_time))))
    per_fold_times = 'Times per fold {}'.format(per_fold_times)
    prediction_test_data = prediction_test_data + total_time_str + '\n' + per_fold_times + '\n'

    return all_predictions, total_time_str, per_fold_times, prediction_test_data
