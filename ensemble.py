from os import listdir, chdir
import torch
from data import build_dataloader, device

train_dataloader, test_dalaloader, val_dataloader = build_dataloader()

def sum_ans(models, x, coefs):
	return (models[i](x) * coefs[i] for i in range(len(models))).sum(dim = 0)

def train(coefs, opt, dataloader, models, fun_loss):
	for model in models:
		model.eval()
	for x, y in dataloader:
		opt.zero_grad()
		x = x.to(device)
		y = y.to(device)
		ans = sum_ans(models, x, coefs)
		loss = fun_loss(ans, y)
		loss.backward()
		opt.step()

chdir('models')
models = []
for file in listdir():
	print(file[-6:])
	if file[-6:] != ".pt.pt":
		continue
	print(file)
	models.append(torch.load(file), map_location=torch.device('cpu'))

coefs = [torch([1. / len(models)], requires_grad=True) for model in models]

opt = torch.optim.Adam(coefs, lr=1e-3)
loss = nn.CrossEntropyLoss()

for i in range(100):
	print(i)
	train(coefs, opt, train_dataloader, models, loss)
