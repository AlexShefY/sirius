import neptune.new as neptune
import torch
import pickle
from os import listdir
from torch.utils.data import DataLoader
run = neptune.init(
    project="lora0207/sirius",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==",
)

project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")

config = "train one model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataloader():
    if 'train_data_v.bin.bin' not in listdir():
      project['train_data_v.bin'].download()
    if 'test_data_v.bin.bin' not in listdir():
      project['test_data_v.bin'].download()
    if 'val_data_v.bin.bin' not in listdir():
      project['val_data_v.bin'].download()

    with open('train_data_v.bin.bin', 'rb') as file:
        train_data = pickle.load(file)

    with open('test_data_v.bin.bin', 'rb') as file:
        test_data = pickle.load(file)

    with open('val_data_v.bin.bin', 'rb') as file:
        val_data = pickle.load(file)

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    return train_dataloader, val_dataloader, test_dataloader