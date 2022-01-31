import torch
import pickle
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from qhoptim.pyt  import QHAdam

from torch.optim.lr_scheduler import ExponentialLR

import neptune.new as neptune

def write_solution(filename, labels):
    with open(filename, 'w') as solution:
        print('Id,Category', file=solution)
        for i, label in enumerate(labels):
            print(f'{i},{label}', file=solution)
            
project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")

with open('test_data_v.bin.bin', 'rb') as file:
    test_data = pickle.load(file)

test_dataloader = DataLoader(test_data, batch_size=512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = resnet18()

name = 'models/model716479_31.pt.pt'

model = torch.load(name, map_location=torch.device('cpu'))

#print(len(test_dataloader))
predictions = []

from tqdm import tqdm

print(len(test_dataloader.dataset))
model.eval()
i = 0
with torch.no_grad():
    for X, _ in tqdm(test_dataloader):
        X = X.to(device)
        pred = model(X).argmax(1).cpu().numpy()
        predictions.extend(list(pred))

print("done")
write_solution('solution.csv', predictions)
