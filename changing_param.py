import torch
import pickle
from models import M5, resNet, denseNet, CnnFnnModel, ModifiedResNet, CnnFnnModel_deeper, mobileNet
from train_one_model import def_train_one_model
from different_param import def_different_param

import optuna
from data import project, run, device
from optuna.visualization import plot_optimization_history

from work_model import test
from data import build_dataloader

import neptune.new as neptune
train_dataloader, val_dataloader, test_dataloader = build_dataloader()

model = mobileNet()
print(model)
model = model.to(device) 

from torch import nn

def get_accuracy(trial):
  params_optim = {
    'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
    'nus_first': trial.suggest_float('nus_first', 0.4, 1.0, log=True),
    'nus_second': trial.suggest_float('nus_second', 0.8, 1.0, log=True),
    'betas_first': trial.suggest_float('betas_first', 0.9, 1.0, log=True),
    'betas_second': trial.suggest_float('betas_second', 0.9, 1.0, log=True),
    'gamma': trial.suggest_float('gamma', 0.8, 1.0, log=True),
    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-5, log=True)
  }
  params_change = {
    'brightness': trial.suggest_float('brightness', 0.1, 0.3, log=True),
    'contrast': trial.suggest_float('contrast', 0.1, 0.3, log=True),
    'hue': trial.suggest_float('hue', 0.1, 0.3, log=True),
    'distortion_scale': trial.suggest_float('distortion_scale', 0.3, 0.7, log=True),
    'p': trial.suggest_float('p', 0.3, 0.7, log=True),
    'saturation': trial.suggest_float('saturation', 0.1, 0.3, log=True)
  }
  global model
  model1 = model
  loss = nn.CrossEntropyLoss()
  acc1 = test(val_dataloader, 0, model, loss, False)[0]
  cur_acc = def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader, params_optim, params_change, epochs=5, flag=True, start_epoch = trial.number * 5)
  if cur_acc < acc1:
    model = model1
  return cur_acc

import optuna

study = optuna.create_study()
study.optimize(get_accuracy, n_trials=10)

print("best params:", study.best_params)

plot_optimization_history(study)
