import json
import datetime as dt
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import ruamel.yaml as yaml

import torch
import torch.distributed as dist

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


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


def save_on_master(*args, **kwargs):
    if get_rank() == 0:
        torch.save(*args, **kwargs)


def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)


def write_json(data, wpath):
    with open(wpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_jsonl(rpath):
    data = []
    with open(rpath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data, wpath, option):
    with open(wpath, option) as f:
        for sample in data:
            f.write(json.dumps(sample)+'\n')


def write_txt_doc(data, wpath):
    with open(wpath, 'a') as f:
        #f.write(f'File used: {test_file} \t Output: {file} \t Random seed: {seed}\n')
        f.write(data)


def write_jsonl_per_fold(fold_i, fold_preds, output_path):
    fold_out = output_path.replace(".jsonl", f"_fold{fold_i}.jsonl")
    write_jsonl(fold_preds, fold_out, 'w')
    print("### Fold results saved to:", fold_out, flush=True)
    return fold_out


def create_test_file(wpath, append, number_start, number_finish, dataset, image_path, answer, question = "Is there a fire in this image?"):
    data = []
    answer_list = []
    answer_list.append(answer)
    for number in range(number_start, number_finish+1):
        image = f"{image_path}{number}.jpg"
        data.append({"dataset": dataset, "question": question, "answer": answer_list, "index": number, "image": image})
    if append:
        write_jsonl(data, wpath, 'a')
    else:
        write_jsonl(data, wpath, 'w')


def update_config(config, override_cfg_str=""):
    if override_cfg_str != "":
        override_cfg_str = override_cfg_str.replace(";", "\n").replace(":", ": ")
        override_cfg = yaml.load(override_cfg_str, Loader=yaml.Loader)
        for k, v in override_cfg.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    config[k][kk] = vv
            else:
                config[k] = v

@dataclass
class SummaryReport:
    test_file: str
    seed: int
    dataset_size: int
    num_threads: int
    cross_val: bool = False
    mpi: bool = False
    folds: int = 0
    world_size: int = 0
    num_threads: int = 0
    NAMES = ["accuracy", "f1_fire", "f1_nofire", "f1_macro", "mcc_fire", "mcc_nofire"]
    LABELS = {
        "accuracy": "Accuracy",
        "f1_fire": "F1 fire",
        "f1_nofire": "F1 nofire",
        "f1_macro": "F1 macro",
        "mcc_fire": "Fire MCC",
        "mcc_nofire": "No Fire MCC",
    }

    @staticmethod
    def time(seconds):
        return str(dt.timedelta(seconds=int(seconds)))

    @staticmethod
    def metric(value):
        return "N/A" if value is None else f"{value:.4f}"

    @classmethod
    def rank_times(cls, rank_times):
        """rank_times: list (index = rank) or dict {rank: seconds}."""
        if not rank_times:
            return ""
        items = rank_times.items() if isinstance(rank_times, dict) else enumerate(rank_times)
        return "Rank times: " + ", \t ".join(f"Rank {r}={cls.time(t)}" for r, t in items)
    
    def _metrics_block(self, metrics):
        return (
            f"Accuracy={self.metric(metrics['accuracy'])}, "
            f"F1 fire={self.metric(metrics['f1_fire'])}, "
            f"F1 nofire={self.metric(metrics['f1_nofire'])}, "
            f"F1 macro={self.metric(metrics['f1_macro'])}, "
            f"Fire MCC={self.metric(metrics.get('mcc_fire'))}, \t "
            f"No Fire MCC={self.metric(metrics.get('mcc_nofire'))}, \t "
        )
    
    def _header(self, output_file, elapsed_s, dataset_time=0, process_time=0, rank_times=None):
        header = f"File used: {self.test_file} \t Output: {output_file} \t Random seed: {self.seed}\n"
        if self.cross_val or self.mpi:
            header += (
                f"Dataset size={self.dataset_size}, \t Number of folds={self.folds}, \t "
                f"Number of Processes={self.world_size}, \t Number of Threads={self.num_threads}\n"
            )
        header += f"Time={self.time(elapsed_s)} \t Time used with IO={dataset_time}, \t Processing time={process_time}\n"
        if rank_times:
            header += "\n" + self.rank_times(rank_times)
        return header

    def aggregate(self, all_fold_reports):
        """Return {'<metric>_mean': ..., '<metric>_std': ...} across folds."""
        aggregate = {}
        for name in self.NAMES:
            values = [fr["metrics"][name] for fr in all_fold_reports
                      if fr["metrics"].get(name) is not None]
            if not values:
                continue
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values))
        return aggregate

    def fold_line(self, fold_id, train_size, val_size, train_loss, elapsed_s, metrics):
        """One line per fold."""
        return (
            f"Fold {fold_id}: train={train_size}, \t val={val_size}, \t loss={train_loss:.4f}, \t Time={self.time(elapsed_s)}"
            + self._metrics_block(metrics)
        )

    def full(self, output_file, metrics, elapsed_s, rank_times=None):
        # Remember to add peak cpu or gpu usage
        """Full report: header, metrics, total time, then each rank's time."""
        report = self._header(output_file, elapsed_s, rank_times) + self._metrics_block(metrics)
        return report

    def short(self, elapsed_s, aggregate=None, metrics=None):
        """Short report for printing: number of images, total time, average metrics.
            Pass `aggregate` (from MetricsAggregator.aggregate) for k-fold runs,
            or `metrics` for a single run.
        """
        parts = [f"Images={self.dataset_size}", f"Total time={self.time(elapsed_s)}"]
        for name in self.NAMES:
            label = self.LABELS[name]
            if aggregate and f"{name}_mean" in aggregate:
                parts.append(f"{label}={aggregate[f'{name}_mean']:.4f} ± {aggregate[f'{name}_std']:.4f}")
            elif metrics and metrics.get(name) is not None:
                parts.append(f"{label}={metrics[name]:.4f}")
        return " | ".join(parts)