from os import listdir, chdir
import os
import torch
from torch import nn
from data import build_dataloader, device, run
from tqdm import tqdm

train_dataloader, val_dataloader, test_dataloader = build_dataloader()

from run_test import write_solution

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

def test(coefs, dataloader, models, step):
  accur = 0
  cnt = 0
  for x, y in dataloader:
    x = x.to(device)
    y = y.to(device)
    cnt += x.shape[0]
    accur += (sum_ans(models, x, coefs).argmax(dim=1) == y).type(torch.float).sum()
  accur /= cnt
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


models = get_models('models')
print(os.getcwd())
coefs = get_random(val_dataloader, models, 10)
write_solution('solution.csv', get_predictions(models, coefs))