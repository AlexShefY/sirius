from os import listdir, chdir
import os
import torch
from torch import nn
from data import build_dataloader, device, run
from tqdm import tqdm
from download_model import all_models
train_dataloader, val_dataloader, test_dataloader = build_dataloader()

from write_solution import write_solution

import random

params_change = {
    'brightness': random.uniform(0.1, 0.3),
    'contrast': random.uniform(0.1, 0.3),
    'hue': random.uniform(0.1, 0.3),
    'distortion_scale': random.uniform(0.4, 0.7),
    'p': random.uniform(0.4, 0.7),
    'saturation': random.uniform(0.1, 0.3) 
}

from torchvision.transforms import ColorJitter, RandomPerspective


jitter = ColorJitter(brightness=params_change['brightness']
    , contrast=params_change['contrast']
    , saturation=params_change['saturation'], hue=params_change['hue']
    )
perspective = RandomPerspective(params_change['distortion_scale'], params_change['p'])

def sum_ans(models, x, coefs):
	return torch.stack([models[i](jitter(perspective(x.to(device)))) * coefs[i].to(device) for i in range(len(models))]).sum(dim = 0)

def train(coefs, opt, dataloader, models, fun_loss):
	for model in models:
		model.eval()
	c = 0
	for x, y in dataloader:
		opt.zero_grad()
		x = x.to(device)
		y = y.to(device)
		ans = sum_ans(models, x, coefs)
		loss = fun_loss(ans, y)
		c += 1
		if c % 100 == 0:
			print(c)
		loss.backward()
		opt.step()

def test(coefs, dataloader, models, step, flag=True):
  accur = 0
  cnt = 0
  for x, y in dataloader:
    x = x.to(device)
    y = y.to(device)
    cnt += x.shape[0]
    accur += (sum_ans(models, x, coefs).argmax(dim=1) == y).type(torch.float).sum()
  accur /= cnt
  if flag:
    run['accuracy'].log(accur, step=step)
  return accur

def get_random(dataloader, models, iters):
	best_coefs = [torch.tensor([0.125]) for model in models]
	best_accur = test(best_coefs, dataloader, models, 0)
	for i in tqdm(range(iters)):
		coefs = torch.rand(len(models))
		accur = test(coefs, dataloader, models, i + 1)
		if accur > best_accur:
			best_accur = accur
			best_coefs = coefs
	return best_coefs

def get_trained(train_dataloader, val_dataloader, models, initial_coefs, iters):
	coefs = initial_coefs
	opt = torch.optim.Adam(coefs, lr=1e-3)
	loss = nn.CrossEntropyLoss()
	test(coefs, train_dataloader, models, 0)
	for i in tqdm(range(iters)):
		train(coefs, opt, train_dataloader, models, loss)
		test(coefs, val_dataloader, models, i + 1)
	return coefs

def get_models(directory):
	chdir(directory)
	models = []
	files = []
	for file in listdir():
		if file[-3:] != ".pt":
			continue
		files.append(file)
		models.append(torch.load(file, map_location=device))
	for model in models:
		model.eval()
	chdir('../')
	return models


def get_predictions(models, coefs):
	predictions = []
	with torch.no_grad():
		for X, _ in test_dataloader:
			X = X.to(device)
			pred = sum_ans(models, X, coefs).argmax(dim=1).cpu().numpy()
			predictions.extend(list(pred))
	return predictions

import optuna
from math import log

class get_cost(object):
  def __init__(self, models):
    self.models = models
  def __call__(self, trial):
    coefs = []
    for i, model in enumerate(self.models):
      coefs.append(torch.tensor([trial.suggest_float(f'{i}_coef', 0.1, 1, log=True)]))
    test(coefs, val_dataloader, self.models, trial.number, True)
    return -log(test(coefs, train_dataloader, self.models, trial.number, False))

def get_trial(models):
  study = optuna.create_study()
  study.optimize(get_cost(models), n_trials = 50)
  coefs = study.best_params
  return [torch.tensor(coefs[x]) for x in coefs.keys()]

all_models()
models = get_models('models')
coefs = get_trial(models)
write_solution('solution.csv', get_predictions(models, coefs))