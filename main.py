import torch
import pickle
from models import M5, resNet, denseNet, CnnFnnModel, ModifiedResNet, M4, CnnFnnModel_deeper
from train_one_model import def_train_one_model

from data import project, run, device

from data import build_dataloader

train_dataloader, val_dataloader, test_dataloader = build_dataloader()

from torch import nn

model = M4()
model = model.to(device)

print(model)

params_optim = {
    'lr': 1e-3,
    'nus_first': 0.38,
    'nus_second': 1.0,
    'betas_first': 0.9,
    'betas_second': 0.99,
    'gamma': 0.9,
    'weight_decay': 1e-6
}

import random 

params_change = {
    'brightness': random.uniform(0.1, 0.3),
    'contrast': random.uniform(0.1, 0.3),
    'hue': random.uniform(0.1, 0.3),
    'distortion_scale': random.uniform(0.4, 0.7),
    'p': random.uniform(0.4, 0.7),
    'saturation': random.uniform(0.1, 0.3) 
}

def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader, params_optim, params_change)
