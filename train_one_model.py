from work_model import train, test
from qhoptim.pyt  import QHAdam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch

from data import project, run, config, device

def def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader):
#	print(model)
	optim =  QHAdam(model.parameters(), lr=4e-3, nus = (0.7, 1.0), betas=(0.995, 0.999))

	scheduler = ExponentialLR(optimizer = optim, gamma = 0.97)

	epochs = 50

	loss = nn.CrossEntropyLoss()
	import time
	import random 

	random.seed(time.time())
	t = random.randint(1, 1000000)
	print(t)
	for epoch in range(epochs):
	    train(train_dataloader, epoch + 1, model, optim, loss)
	    test(val_dataloader,epoch + 1, model, loss)
	    scheduler.step()
	    torch.save(model, f'model{t}_{epoch}.pt')
	    run[f'model{t}_{epoch}.pt'].upload(f'model{t}_{epoch}.pt')