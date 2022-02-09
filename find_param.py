import torch
import pickle
from models import M5, resNet, denseNet, CnnFnnModel, ModifiedResNet
from train_one_model import def_train_one_model
from different_param import def_different_param

import optuna
from data import project, run, config, device
from optuna.visualization import plot_optimization_history

from data import build_dataloader
from data import project, run, config, device, lr, nus_first, nus_second, betas_first, betas_second, gamma

train_dataloader, val_dataloader, test_dataloader = build_dataloader()

def get_accuracy(trial):
  model = CnnFnnModel()
  model = model.to(device)
  lr = trial.suggest_float('lr', 1e-4, 1e-1)
  nus_first = trial.suggest_float('nus_first', 0.1, 1.0)
  nus_second = trial.suggest_float('nus_second', 0.1, 1.0)
  betas_first = trial.suggest_float('betas_first', 0.9, 1.0)
  betas_second = trial.suggest_float('betas_second', 0.9, 1.0)
  gamma = trial.suggest_float('gamma', 0.8, 1.0)
  return def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader, epochs=5, flag=False)

import optuna

study = optuna.create_study()
study.optimize(get_accuracy, n_trials=10)

print(study.best_params)

plot_optimization_history(study)
