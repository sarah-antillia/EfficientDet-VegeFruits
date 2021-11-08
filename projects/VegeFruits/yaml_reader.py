import yaml

file = "./configs/hparams.yaml"

with open(file, 'r') as f:
    data = yaml.safe_load(f)
    print(data) # 