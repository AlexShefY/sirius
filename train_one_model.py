from work_model import train, test
from qhoptim.pyt  import QHAdam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch

from data import project, run, config, device

def def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader):
	run['lr'] = lr
	run['nus_first'] = nus_first
	run['nus_second'] = nus_second
	run['betas_first'] = betas_first
	run['betas_second'] = betas_second
	run['gamma'] = gamma
	optim =  QHAdam(model.parameters(), lr=ls, nus = (nus_first, nus_second), betas=(betas_first, betas_second))

	scheduler = ExponentialLR(optimizer = optim, gamma = gamma)

	epochs = 50

	loss = nn.CrossEntropyLoss()
	import time
	import random 

	random.seed(time.time())
	t = random.randint(1, 1000000)
	for epoch in range(epochs):
	    train(train_dataloader, epoch, model, optim, loss)
	    test(val_dataloader,epoch + 1, model, loss)
	    scheduler.step()
	    torch.save(model, f'models_rubbish/model{t}_{epoch + 1}.pt')
	    run[f'models_rubbish/model{t}_{epoch + 1}.pt'].upload(f'model{t}_{epoch + 1}.pt')