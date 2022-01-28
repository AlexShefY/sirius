import torch
from work_model import train, test
from qhoptim.pyt  import QHAdam
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn

from torchvision.models import resnet18
from data import project, run, config, device

def def_different_param(train_dataloader, val_dataloader, test_dataloader, lr_l, nus_first_l, nus_second_l, betas_first_l, betas_second_l):
  
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
      train(train_dataloader, epoch + 1, model, optim, loss, False)
      scheduler.step()
      
    (accur, sum_loss) = test(val_dataloader, epoch + 1, model, loss, False)
    
    run['accur'].log(accur, step=i)
    run['sum_loss'].log(sum_loss, step=i)
    run['lr'].log(lr_l[i], step=i)
    run['nus_first'].log(nus_first_l[i], step=i)
    run['nus_second'].log(nus_second_l[i], step=i)
    run['betas_first'].log(betas_first_l[i], step=i)
    run['betas_second'].log(betas_second_l[i], step=i)