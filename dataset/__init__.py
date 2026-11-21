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

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.transforms import InterpolationMode


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


def create_dataset(dataset, config):
    if dataset == 'eval':
        from dataset.eval_datasets import LynxEvalDatasetBase
        test_dataset = LynxEvalDatasetBase(config)
        return test_dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")

def create_train_dataset(dataset, config):
    if dataset == 'eval':
        from dataset.train_datasets import LynxTrainDatasetBase
        train_dataset = LynxTrainDatasetBase(config)
        return train_dataset
    else:
        raise NotImplementedError(f"dataset == {dataset}")


def create_loader(datasets, batch_size, num_workers, collate_fns, shuffles, pin_memories):
    loaders = []
    for dataset, bs, n_worker, collate_fn, shuffle, pin_memory in zip(datasets, batch_size, num_workers, collate_fns, shuffles, pin_memories):
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
        loaders.append(loader)

    return loaders
