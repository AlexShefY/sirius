from work_model import train, test
from qhoptim.pyt  import QHAdam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch
from tqdm import tqdm
from math import log

from data import project, run, config, device

def def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader, params_optim, params_change, epochs=50, flag=True, start_epoch=-0):
	run['lr'] = params_optim['lr']
	run['nus_first'] = params_optim['nus_first']
	run['nus_second'] = params_optim['nus_second']
	run['betas_first'] = params_optim['betas_first']
	run['betas_second'] = params_optim['betas_second']
	run['gamma'] = params_optim['gamma']
	run['weight_decay'] = params_optim['weight_decay']
	print("params:", params_optim['lr'], params_optim['nus_first'], params_optim['nus_second'],
  params_optim['betas_first'], params_optim['betas_second'], params_optim['gamma'])
	optim =  QHAdam(model.parameters(), lr=params_optim['lr'], nus = (params_optim['nus_first'], params_optim['nus_second']),
   betas=(params_optim['betas_first'], params_optim['betas_second']), weight_decay=params_optim['weight_decay'])

	scheduler = ExponentialLR(optimizer = optim, gamma = params_optim['gamma'])

	loss = nn.CrossEntropyLoss()
	import time
	import random 

	random.seed(time.time())
	t = random.randint(1, 1000000)
	last_val = 0
	for epoch in tqdm(range(epochs)):
	    train(train_dataloader, start_epoch + epoch, model, optim, loss, params_change, flag)
	    last_val = test(val_dataloader, start_epoch + epoch + 1, model, loss, flag)[0]
	    scheduler.step()
	    torch.save(model, f'models_rubbish/model{t}_{start_epoch + epoch + 1}.pt')
	    run[f'models_rubbish/model{t}_{start_epoch + epoch + 1}.pt'].upload(f'models_rubbish/model{t}_{start_epoch + epoch + 1}.pt')
	return -log(last_val)