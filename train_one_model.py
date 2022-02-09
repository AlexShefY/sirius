from work_model import train, test
from qhoptim.pyt  import QHAdam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch
from tqdm import tqdm
from math import log

from data import project, run, config, device, lr, nus_first, nus_second, betas_first, betas_second, gamma

def def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader, epochs=50, flag=True):
	run['lr'] = lr
	run['nus_first'] = nus_first
	run['nus_second'] = nus_second
	run['betas_first'] = betas_first
	run['betas_second'] = betas_second
	run['gamma'] = gamma
	optim =  QHAdam(model.parameters(), lr=lr, nus = (nus_first, nus_second), betas=(betas_first, betas_second))

	scheduler = ExponentialLR(optimizer = optim, gamma = gamma)

	loss = nn.CrossEntropyLoss()
	import time
	import random 

	random.seed(time.time())
	t = random.randint(1, 1000000)
	for epoch in tqdm(range(epochs)):
	    train(train_dataloader, epoch, model, optim, loss, flag)
	    test(val_dataloader,epoch + 1, model, loss, flag)
	    scheduler.step()
	    torch.save(model, f'models_rubbish/model{t}_{epoch + 1}.pt')
	    run[f'models_rubbish/model{t}_{epoch + 1}.pt'].upload(f'models_rubbish/model{t}_{epoch + 1}.pt')
	return -log(test(val_dataloader, epochs + 1, model, loss, flag)[0])