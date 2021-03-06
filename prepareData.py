import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from IPython.display import clear_output

import os
import pickle
from typing import Any, Callable, Optional, Tuple
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose
from torchvision.datasets.vision import VisionDataset


import neptune.new as neptune

project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")


class SVHN(VisionDataset):

    def __init__(self,
                 root: str,
                 is_train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.is_train = is_train

        meta_path = os.path.join(self.root, 'meta')
        with open(meta_path, "rb") as f:
            content = pickle.load(f)
            self.classes = content['label_names']
            self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        data_path = os.path.join(self.root, 'data_train' if is_train else 'data_test')
        with open(data_path, "rb") as f:
            content = pickle.load(f)
            self.data = content['images']
            self.targets = content.get('labels')

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.fromarray(self.data[index].astype(np.uint8))
        target = self.targets[index] if self.is_train else len(self.classes)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)
        
    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

data = SVHN(
    root="sirius-spbsu-2022-entry-competition",
    is_train=True,
    transform=ToTensor(),
)

train_data, val_data = torch.utils.data.random_split(
    data, 
    [40000, 10000], 
    generator=torch.Generator().manual_seed(137),
)

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

test_data = SVHN(
    root="sirius-spbsu-2022-entry-competition",
    is_train=False,
    transform=ToTensor(),
)

test_dataloader = DataLoader(test_data, batch_size=batch_size,)
import neptune.new as neptune
from neptune.new.types import File

project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")

lst1 = [(x, y) for (x, y) in train_dataloader.dataset]
lst2 = [(x, y) for (x, y) in val_dataloader.dataset]
lst3 = [(x, y) for (x, y) in test_dataloader.dataset]

import plotly.express as px
import numpy as np

def smooth(mat):
    sum_ = torch.zeros_like(mat)
    one = torch.ones_like(mat)
    cnt_ = torch.zeros_like(mat)
    b = (0 < mat) & (mat < 1)
    g = torch.where(b, mat, sum_)
    g1 = torch.where(b, one, sum_)
    cnt_[:, 1:, :] += g1[:, :-1, :]
    cnt_[:, :-1, :] += g1[:, 1:, :]
    cnt_[:, :, 1:] += g1[:, :, :-1]
    cnt_[:, :, :-1] += g1[:, :, 1:]
    sum_[:, 1:, :] += g[:, :-1, :]
    sum_[:, :-1, :] += g[:, 1:, :]
    sum_[:, :, 1:] += g[:, :, :-1]
    sum_[:, :, :-1] += g[:, :, 1:]
    b3 = (b == 0)
    b2 = b3 + cnt_
    b1 = (b2 < 2)
    return torch.where(b1, mat, sum_ / cnt_)

from show_images import im_show

def prepare_dataset(dataset):
    for i in range(len(dataset)):
        for t in range(3):
            dataset[i] = (smooth(dataset[i][0]), dataset[i][1])
    return dataset

import random

def show_random(dataset, cnt):
    for j in range(cnt):
        i = random.randint(0, 10000)
        im_show(lst1[i][0])

import pickle

def upload(dataset, file):
    fl = open(file, "wb")
    pickle.dump(dataset, fl)
    project[file].upload(file)

lst1 = prepare_dataset(lst1)
lst2 = prepare_dataset(lst2)
lst3 = prepare_dataset(lst3)

show_random(lst1, 2)
show_random(lst2, 2)
show_random(lst3, 2)

upload(lst1, "train_data_v1.bin")
upload(lst2, "val_data_v1.bin")
upload(lst3, "test_data_v1.bin")

print("Done")