#!pip install neptune-client qhoptim
import torch
import pickle
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from qhoptim.pyt  import QHAdam

from torch.optim.lr_scheduler import ExponentialLR

import neptune.new as neptune

run = neptune.init(
    project="lora0207/sirius",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==",
)

project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")

#project['train_data_v.bin'].download()
#project['test_data_v.bin'].download()
#project['val_data_v.bin'].download()

with open('train_data_v.bin.bin', 'rb') as file:
    train_data = pickle.load(file)

with open('test_data_v.bin.bin', 'rb') as file:
    test_data = pickle.load(file)

with open('val_data_v.bin.bin', 'rb') as file:
    val_data = pickle.load(file)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

from torch import nn

def train(steps, model, optim, fun_loss):
    model.train()
  #  i = 0
    for batch, (x, y) in enumerate(train_dataloader):
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        ans = model(x)
    #    print(x.shape)
    #    print(y.shape)
    #    print(ans.shape)
    #    print(y.dtype)
    #    print(ans.dtype)
        loss = fun_loss(ans, y)
    #    print(loss)
        #print(fun_loss(ans, torch.ones(64, 10)))
        loss.backward()
        optim.step()
        step =  steps + (1 + batch) / len(train_dataloader)
        if batch % 1 == 0:
            run['losses'].log(loss.item(), step=step)
       # scheduler.step()




def test(step, model, fun_loss):
    model.eval()
    accur = 0
    sum_loss = 0
    cnt = 0 
    with torch.no_grad():
      for x, y in val_dataloader:
          x = x.to(device)
          y = y.to(device)
          ans = model(x)
          loss = fun_loss(ans, y)
          #print(loss)
          #print(fun_loss(ans, torch.ones(64, 10)))
          #print(loss.item())
          sum_loss += loss.item()
          cnt += x.shape[0]
          accur += (ans.argmax(dim=1) == y).type(torch.float).sum().item()

    accur /= cnt
    print(sum_loss)
    sum_loss /= len(val_dataloader)
    print(accur, sum_loss)
    run['losses_test'].log(sum_loss, step=step)
    run['accuracy'].log(accur, step=step)

model = resnet18()
model.fc = nn.Linear(512, 10)
model = model.to(device)
#optim = torch.optim.Adam(model.parameters(), lr=1e-3)

import plotly.express as px
import numpy as np

def im_show(pic, i):
    print(pic.shape)
    to_plot = 32 * [[[]]]
    for a in range(32):
        to_plot[a] = 32 * [[]]
        for b in range(32):
            to_plot[a][b] = 3 * [0]
            to_plot[a][b][0] = pic[0][a][b].item()
            to_plot[a][b][1] = pic[1][a][b].item()
            to_plot[a][b][2] = pic[2][a][b].item()
    fig = px.imshow(to_plot)
    fig.show()
    #fig.save("im${i}.jpg")

optim =  QHAdam(model.parameters(), lr=1e-3, nus = (0.7, 1.0), betas=(0.995, 0.999))

scheduler = ExponentialLR(optimizer = optim, gamma = 0.95)

#print("A")
#for i in range(10):
#    im_show(train_dataloader.dataset[i + 253][0], i)
#    print(train_dataloader.dataset[i + 253][1])
epochs = 50

loss = nn.CrossEntropyLoss()

import time
import random 

random.seed(time.time())
t = random.randint(1, 1000000)
print(t)
for epoch in range(epochs):
    train(epoch + 1, model, optim, loss)
    test(epoch + 1, model, loss)
    scheduler.step()
    torch.save(model, f'model{t}_{epoch}.pt')
    run[f'model{t}_{epoch}.pt'].upload(f'model{t}_{epoch}.pt')