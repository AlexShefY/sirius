import neptune.new as neptune

def one_model(run_nunmber, model_name):
    run_ = neptune.init(
        project='lora0207/sirius',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==',
        run=f'SIR-{run_number}' # for example 'SAN-123'
        )

    run_[f'models_rubbish/{model_name}.pt'].download('models')

def all_models(run_number):
    run_ = neptune.init(
        project='lora0207/sirius',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZmQyMjc4Ni02NWQwLTRiZTYtYWIyZC0yOGJjOTE2NDNmODEifQ==',
        run=f'SIR-{run_number}' # for example 'SAN-123'
        )

    run_[f'models'].download()

if __name__ == "__main__":
    all_models()