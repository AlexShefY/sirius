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


# class SVHN(VisionDataset):

#     def __init__(self,
#                  root: str,
#                  is_train: bool = True,
#                  transform: Optional[Callable] = None,
#                  target_transform: Optional[Callable] = None,
#                  ) -> None:

#         super().__init__(root, transform=transform, target_transform=target_transform)
#         self.is_train = is_train

#         meta_path = os.path.join(self.root, 'meta')
#         with open(meta_path, "rb") as f:
#             content = pickle.load(f)
#             self.classes = content['label_names']
#             self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

#         data_path = os.path.join(self.root, 'data_train' if is_train else 'data_test')
#         with open(data_path, "rb") as f:
#             content = pickle.load(f)
#             self.data = content['images']
#             self.targets = content.get('labels')

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         img = Image.fromarray(self.data[index].astype(np.uint8))
#         target = self.targets[index] if self.is_train else len(self.classes)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target

#     def __len__(self) -> int:
#         return len(self.data)
        
#     def extra_repr(self) -> str:
#         split = "Train" if self.train is True else "Test"
#         return f"Split: {split}"

# data = SVHN(
#     root="sirius-spbsu-2022-entry-competition",
#     is_train=True,
#     transform=ToTensor(),
# )

# train_data, val_data = torch.utils.data.random_split(
#     data, 
#     [40000, 10000], 
#     generator=torch.Generator().manual_seed(137),
# )

# batch_size = 64
# train_dataloader = DataLoader(train_data, batch_size=batch_size)
# val_dataloader = DataLoader(val_data, batch_size=batch_size)

# test_data = SVHN(
#     root="sirius-spbsu-2022-entry-competition",
#     is_train=False,
#     transform=ToTensor(),
# )

# test_dataloader = DataLoader(test_data, batch_size=batch_size,)
import neptune.new as neptune
from neptune.new.types import File

# project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")

# lst1 = [(x, y) for (x, y) in train_dataloader.dataset]
# lst2 = [(x, y) for (x, y) in val_dataloader.dataset]
# lst3 = [(x, y) for (x, y) in test_dataloader.dataset]

# import plotly.express as px
# import numpy as np

# dx = [-1, 0, 0, 1]
# dy = [0, 1, -1, 0]
# s = 3 * [0]
# cnt = 3 * [0]
# to_plot = 32 * [[[]]]
# for x in range(32):
#     to_plot[x] = 32 * [[]]
#     for y in range(32):
#         to_plot[x][y] = 3 * [0]

# import cv2 as cv

# def smooth(mat):
#     sum_ = torch.zeros_like(mat)
#     one = torch.ones_like(mat)
#     cnt_ = torch.zeros_like(mat)
#     b = (0 < mat) & (mat < 1)
#     g = torch.where(b, mat, sum_)
#     g1 = torch.where(b, one, sum_)
#     cnt_[:, 1:, :] += g1[:, :-1, :]
#     cnt_[:, :-1, :] += g1[:, 1:, :]
#     cnt_[:, :, 1:] += g1[:, :, :-1]
#     cnt_[:, :, :-1] += g1[:, :, 1:]
#     sum_[:, 1:, :] += g[:, :-1, :]
#     sum_[:, :-1, :] += g[:, 1:, :]
#     sum_[:, :, 1:] += g[:, :, :-1]
#     sum_[:, :, :-1] += g[:, :, 1:]
#     return torch.where(b, mat, sum_ / cnt_)

# for i in range(len(lst1)):
#     lst1[i] = (smooth(lst1[i][0]), lst1[i][1])
#     lst1[i] = (smooth(lst1[i][0]), lst1[i][1])
#     lst1[i] = (smooth(lst1[i][0]), lst1[i][1])



# for i in range(len(lst3)):
#     lst3[i] = (smooth(lst3[i][0]), lst3[i][1])
#     lst3[i] = (smooth(lst3[i][0]), lst3[i][1])
#     lst3[i] = (smooth(lst3[i][0]), lst3[i][1])

# for j in range(10):
#     i = j + 97
#     for x in range(32):
#        for y in range(32):
#         to_plot[x][y][0] = lst1[i][0][0][x][y].item()
#         to_plot[x][y][1] = lst1[i][0][1][x][y].item()
#         to_plot[x][y][2] = lst1[i][0][2][x][y].item()

#     fig = px.imshow(to_plot)
#     fig.show()


import pickle


# fl1 = open("train_data_v.bin", "wb")

# fl2 = open("val_data_v.bin", "wb")

# fl3 = open("test_data_v.bin", "wb")

# pickle.dump(lst1, fl1)
# pickle.dump(lst2, fl2)
# pickle.dump(lst3, fl3)

# project['val_data.bin'] = File("val_data.bin")
# project['train_data.bin'] = File("train_data.bin") 
# project['test_data.bin'] = File("test_data.bin")


project['val_data_v.bin'] = File("val_data_v.bin")
project['train_data_v.bin'] = File("train_data_v.bin") 
project['test_data_v.bin'] = File("test_data_v.bin")
