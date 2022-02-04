import torch
import pickle
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from qhoptim.pyt  import QHAdam

from data import build_dataloader
from torch.optim.lr_scheduler import ExponentialLR

import neptune.new as neptune

project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")

train_dataloader, val_dataloader, test_dataloader = build_dataloader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = resnet18()

name = 'models/model923187_24.pt.pt'

model = torch.load(name, map_location=torch.device('cpu'))

#print(len(test_dataloader))
predictions = []

from tqdm import tqdm

print(len(test_dataloader.dataset))
model.eval()

with torch.no_grad():
    for X, _ in test_dataloader:
        X = X.to(device)
        ans = model(X)
        pred = ans.argmax(1).cpu().numpy()
        predictions.extend(list(pred))

write_solution('solution.csv', predictions)
