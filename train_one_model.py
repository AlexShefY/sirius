from work_model import train, test
from qhoptim.pyt  import QHAdam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch
from tqdm import tqdm
from math import log

from data import project, run, config, device

def def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader, params, epochs=50, flag=True, start_epoch=-0):
	run['lr'] = params['lr']
	run['nus_first'] = params['nus_first']
	run['nus_second'] = params['nus_second']
	run['betas_first'] = params['betas_first']
	run['betas_second'] = params['betas_second']
	run['gamma'] = params['gamma']
	run['weight_decay'] = params['weight_decay']
	print("params:", params['lr'], params['nus_first'], params['nus_second'],
  params['betas_first'], params['betas_second'], params['gamma'])
	optim =  QHAdam(model.parameters(), lr=params['lr'], nus = (params['nus_first'], params['nus_second']),
   betas=(params['betas_first'], params['betas_second']), weight_decay=params['weight_decay'])

	scheduler = ExponentialLR(optimizer = optim, gamma = params['gamma'])

	loss = nn.CrossEntropyLoss()
	import time
	import random 

	random.seed(time.time())
	t = random.randint(1, 1000000)
	last_val = 0
	for epoch in tqdm(range(epochs)):
	    train(train_dataloader, start_epoch + epoch, model, optim, loss, flag)
	    last_val = test(val_dataloader, start_epoch + epoch + 1, model, loss, flag)[0]
	    scheduler.step()
	    torch.save(model, f'models_rubbish/model{t}_{start_epoch + epoch + 1}.pt')
	    run[f'models_rubbish/model{t}_{start_epoch + epoch + 1}.pt'].upload(f'models_rubbish/model{t}_{start_epoch + epoch + 1}.pt')
	return -log(last_val)