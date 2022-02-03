from os import listdir, chdir
import torch
from torch import nn
from data import build_dataloader, device, run

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
    #print((sum_ans(models, x, coefs).argmax(dim=1) == y))
    accur += (sum_ans(models, x, coefs).argmax(dim=1) == y).type(torch.float).sum()
  print(accur)
  accur /= cnt
  run['accuracy'].log(accur, step=step)

chdir('models')
models = []
files = []
for file in listdir():
	print(file[-6:])
	if file[-6:] != ".pt.pt":
		continue
	print(file)
	files.append(file)
	models.append(torch.load(file, map_location=device))

coefs = [torch.tensor([0.125], requires_grad=True) for model in models]
#opt = torch.optim.Adam(coefs, lr=1e-3)
#loss = nn.CrossEntropyLoss()

#from tqdm import tqdm

test(coefs, val_dataloader, models, 0)
#for i in tqdm(range(100)):
#	train(coefs, opt, train_dataloader, models, loss)
#	test(coefs, val_dataloader, models, i + 1)

for model in models:
	model.eval()

predictions = []

with torch.no_grad():
    for X, _ in test_dataloader:
        X = X.to(device)
  #      print(X.shape)
  #      for idx, model in enumerate(models):
  #      	print(files[idx])
  #      	if files[idx] == 'model771406_18.pt.pt':
  #      		print(model)
  #      	model(X)
        pred = sum_ans(models, X, coefs).argmax(dim=1).cpu().numpy()
        predictions.extend(list(pred))

write_solution('solution.csv', predictions)