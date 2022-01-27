import torch
import pickle
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from  qhoptim.pyt  import QHAdam

from torch.optim.lr_scheduler import ExponentialLR

import neptune.new as neptune

run = neptune.init(
    project="lora0207/sirius",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==",
)

project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")

project['train_data.bin'].download()
project['test_data.bin'].download()
project['val_data.bin'].download()

with open('train_data.bin.bin', 'rb') as file:
    train_data = pickle.load(file)

with open('test_data.bin.bin', 'rb') as file:
    test_data = pickle.load(file)

with open('val_data.bin.bin', 'rb') as file:
    val_data = pickle.load(file)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

from torch import nn

def train(model, optim, fun_loss):
    model.train()
  #  i = 0
    for batch, (x, y) in enumerate(train_dataloader):
        optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        ans = model(x)
        loss = fun_loss(ans, y)
        #print(fun_loss(ans, torch.ones(64, 10)))
        loss.backward()
        optim.step()
       # scheduler.step()




def test(model, fun_loss):
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

model = resnet18()
model.fc = nn.Linear(512, 10)
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

#optim =  QHAdam(model.parameters(), lr=1e-3, nus = (0.7, 1.0), betas=(0.995, 0.999))

#scheduler = ExponentialLR(optimizer = optim, lr = 1, gamma = 0.99)

epochs = 50

loss = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train(model, optim, loss)
    #print("test")
    #break
    print(hash(str(model.parameters())))
    test(model, loss)