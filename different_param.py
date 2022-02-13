import torch
from work_model import train, test
from qhoptim.pyt  import QHAdam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn

from torchvision.models import resnet18
from data import project, run, device

from data import build_dataloader
train_dataloader, val_dataloader, test_dataloader = build_dataloader()

def def_different_param(train_dataloader, val_dataloader, test_dataloader, lr_l, nus_first_l, nus_second_l, betas_first_l, betas_second_l):

  import random 

  params_change = {
    'brightness': random.uniform(0.1, 0.3),
    'contrast': random.uniform(0.1, 0.3),
    'hue': random.uniform(0.1, 0.3),
    'distortion_scale': random.uniform(0.4, 0.7),
    'p': random.uniform(0.4, 0.7),
    'saturation': random.uniform(0.1, 0.3) 
  }

  for i in range(len(lr_l)):
    model = resnet18()
    model.fc = nn.Linear(512, 10)
    model = model.to(device)
    optim =  QHAdam(model.parameters(), lr=lr_l[i], nus = (nus_first_l[i], nus_second_l[i]),
		betas=(betas_first_l[i], betas_second_l[i]))
    
    scheduler = ExponentialLR(optimizer = optim, gamma = 0.95)
    
    epochs = 5
    
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
      train(train_dataloader, epoch + 1, model, optim, loss, params_change, False)
      scheduler.step()
      
    (accur, sum_loss) = test(val_dataloader, epoch + 1, model, loss, False)
    
    run['accur'].log(accur, step=i)
    run['sum_loss'].log(sum_loss, step=i)
    run['lr'].log(lr_l[i], step=i)
    run['nus_first'].log(nus_first_l[i], step=i)
    run['nus_second'].log(nus_second_l[i], step=i)
    run['betas_first'].log(betas_first_l[i], step=i)
    run['betas_second'].log(betas_second_l[i], step=i)

lr_l = [x * 5e-5 for x in range(1, 20)]
nus_first_l = [0.7] * len(lr_l)
nus_second_l = [1.0] * len(lr_l)
betas_first_l = [0.995] * len(lr_l)
betas_second_l = [0.999] * len(lr_l)
def_different_param(train_dataloader, val_dataloader, test_dataloader, lr_l, nus_first_l, nus_second_l, betas_first_l, betas_second_l)
