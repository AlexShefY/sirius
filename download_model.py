import neptune.new as neptune
import os
from os import listdir

def one_model(run_number, model_name):
    run_ = neptune.init(
        project='lora0207/sirius',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==',
        run=f'SIR-{run_number}' # for example 'SAN-123'
        )
    run_[f'models_rubbish/{model_name}.pt'].download('models')

def all_models():
    if 'models' not in listdir():
      os.makedirs('models')
    project = neptune.init_project(name="lora0207/sirius", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==")
    for model in project.get_structure()['models']:
      if f'models/{model}' not in listdir('models'):
        project[f'models/{model}'].download('models')

if __name__ == "__main__":
    all_models()