from unitraj.datasets.base_dataset import BaseDataset
from unitraj.datasets.simpl_dataset import SimplDataset
from unitraj.datasets.wayformer_dataset import WayformerDataset
from unitraj.datasets.autobot_dataset import AutobotDataset
from unitraj.datasets.MTR_dataset import MTRDataset
from unitraj.datasets.simpl_mae_dataset import SimplMAEDataset

__all__ = [
    'BaseDataset',
    'SimplDataset',
    'WayformerDataset',
    'AutobotDataset',
    'MTRDataset',
    'SimplMAEDataset'
]


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
