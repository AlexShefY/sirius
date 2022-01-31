from os import listdir, chdir
import torch
from torch import nn
from data import build_dataloader, device, run

train_dataloader, val_dataloader, test_dataloader = build_dataloader()

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
for file in listdir():
	print(file[-6:])
	if file[-6:] != ".pt.pt":
		continue
	print(file)
	models.append(torch.load(file, map_location=device))

coefs = [torch.tensor([0.0], requires_grad=True) for model in models]
coefs[0] = torch.tensor([1.0], requires_grad=True)
opt = torch.optim.Adam(coefs, lr=1e-3)
loss = nn.CrossEntropyLoss()

test(coefs, val_dataloader, models, 0)
for i in range(100):
	print(i)
	train(coefs, opt, train_dataloader, models, loss)
	print(coefs)
	test(coefs, val_dataloader, models, i + 1)
