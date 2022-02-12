# import neptune.new as neptune

# run_number = 841
# model_name = 'model418348_34'

# run_ = neptune.init(
#     project='lora0207/sirius',
#     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==',
#     run=f'SIR-{run_number}' # for example 'SAN-123'
#     )

# run_[f'models_rubbish/{model_name}.pt'].download('models')

from torchvision.models import mobilenet_v2

model = mobilenet_v2()
print(model)
