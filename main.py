import torch
import pickle
from torchvision.models import resnet18, resnet50, densenet161
from models import M5
from train_one_model import def_train_one_model
from different_param import def_different_param

from data import project, run, config, device

from data import build_dataloade

train_dataloader, test_dalaloader, val_dataloader = build_dataloader()

from torch import nn

model = M5()
#model = resnet18()
#model.fc = nn.Linear(512, 10)

#for name, module in model.named_children():
#    if name == 'fc':
#        continue
#    print(name, module)
#    module = nn.Sequential(module, nn.Dropout(p=0.1))

model = model.to(device)

print(model)
import plotly.express as px
import numpy as np

def im_show(pic):
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


i = 286
for j in range(10):
    print(train_dataloader.dataset[j + i][0].shape)
    im_show(train_dataloader.dataset[j + i][0])
    print(train_dataloader.dataset[j + i][1])


if config == "train one model":
    def_train_one_model(model, train_dataloader, val_dataloader, test_dataloader)
elif config == "train with different":
    lr_l = [x for x in range(1e-5, 1e-3, 5e-5)]
    nus_first_l = [0.7] * len(lr_l)
    nus_second_l = [1.0] * len(lr_l)
    betas_first_l = [0.995] * len(lr_l)
    betas_second_l = [0.999] * len(lr_l)
    def_different_param(train_dataloader, val_dataloader, test_dataloader, lr_l, nus_first_l, nus_second_l, betas_first_l, betas_second_l)
