from unitraj.models.autobot import AutoBot
from unitraj.models.mtr import MTR
from unitraj.models.wayformer import Wayformer
from unitraj.models.simpl import Simpl
from unitraj.models.simpl_mae import SimplMAEPretrain, SimplMAEFinetune

__all__ = [
    'Simpl',
    'Wayformer',
    'MTR',
    'AutoBot',
    'SimplMAEPretrain',
    'SimplMAEFinetune'
]

def get_model(config):
    model_name = config.method.model_name
    if model_name == 'simpl':
        return Simpl(config.method)
    elif model_name == 'wayformer':
        return Wayformer(config.method)
    elif model_name == 'mtr':
        return MTR(config.method)
    elif model_name == 'autobot':
        return AutoBot(config.method)
    elif model_name == 'simpl-mae-pretrain':
        return SimplMAEPretrain(config.method)
    elif model_name == 'simpl-mae-finetune':
        return SimplMAEFinetune(config.method)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
