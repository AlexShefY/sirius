from os import listdir, chdir
import os
import torch
from torch import nn
from data import build_dataloader, device, run
from tqdm import tqdm

train_dataloader, val_dataloader, test_dataloader = build_dataloader()

from write_solution import write_solution

def sum_ans(models, x, coefs):
	return torch.stack([models[i](x) * coefs[i].to(device) for i in range(len(models))]).sum(dim = 0)

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
		if file[-6:] != ".pt.pt":
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

class get_cost(object):
  def __init__(self, models):
    self.models = models
  def __call__(self, trial):
    coefs = []
    for model, i in enumerate(self.models):
    	coefs.append(torch.tensor([trial.suggest_float(f'i_coef', 0.1, 1, log=True)]))
    return test(coefs, train_dataloader, self.models, trial.number)

def get_trial(models):
  study = optuna.create_study()
  study.optimize(get_cost(models), n_trials = 2)
  return study.best_params

models = get_models('models')
print(len(models))
print(os.getcwd())
coefs = get_trial(models)
coefs1 = [torch.tensor(coefs[x]) for x in coefs.keys()]
print(coefs1)
write_solution('solution.csv', get_predictions(models, coefs1))