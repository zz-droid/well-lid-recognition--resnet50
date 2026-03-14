import yaml
from easydict import EasyDict as edict


def update_config(config_file):
    with open(config_file, encoding='utf-8') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config
