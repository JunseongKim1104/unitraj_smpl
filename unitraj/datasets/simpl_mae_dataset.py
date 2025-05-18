import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

from unitraj.datasets.simpl_dataset import SimplDataset


class SimplMAEDataset(SimplDataset):
    """
    Dataset for SimplMAE model that extends SimplDataset with additional functionality
    for masked autoencoder pretraining.
    """
    def __init__(self, config, split):
        super().__init__(config, split)
        
        # Additional parameters for MAE
        self.is_pretraining = config.get('is_pretraining', True)
        
    def __getitem__(self, idx):
        """
        Get a data sample for the SimplMAE model.
        For pretraining, we need both history and future trajectories.
        For fine-tuning, we use the standard SimplDataset format.
        """
        data = super().__getitem__(idx)
        
        if self.is_pretraining:
            # For pretraining, we need to ensure the future trajectories are included
            # in the input_dict even during training
            data['input_dict']['y'] = data['input_dict']['actors_gt'].permute(0, 2, 1)[..., :2]
        
        return data
    
    def collate_fn(self, batch):
        """
        Collate function for SimplMAE dataset.
        Extends the SimplDataset collate_fn with additional processing for MAE.
        """
        # Use the base collate function
        batch_data = super().collate_fn(batch)
        
        if self.is_pretraining:
            # Additional processing for MAE pretraining
            input_dict = batch_data['input_dict']
            
            # Ensure future trajectories are included
            if 'y' not in input_dict:
                input_dict['y'] = input_dict['actors_gt'].permute(0, 2, 1)[..., :2]
            
            # Add padding masks for lanes and actors
            if 'lane_key_padding_mask' not in input_dict:
                lane_idcs = input_dict['lane_idcs']
                lane_key_padding_mask = torch.zeros(
                    (len(lane_idcs), max([len(idcs) for idcs in lane_idcs])),
                    dtype=torch.bool,
                    device=lane_idcs[0].device
                )
                for i, idcs in enumerate(lane_idcs):
                    if len(idcs) < lane_key_padding_mask.shape[1]:
                        lane_key_padding_mask[i, len(idcs):] = True
                input_dict['lane_key_padding_mask'] = lane_key_padding_mask
            
            if 'x_key_padding_mask' not in input_dict:
                actor_idcs = input_dict['actor_idcs']
                x_key_padding_mask = torch.zeros(
                    (len(actor_idcs), max([len(idcs) for idcs in actor_idcs])),
                    dtype=torch.bool,
                    device=actor_idcs[0].device
                )
                for i, idcs in enumerate(actor_idcs):
                    if len(idcs) < x_key_padding_mask.shape[1]:
                        x_key_padding_mask[i, len(idcs):] = True
                input_dict['x_key_padding_mask'] = x_key_padding_mask
            
            # Add num_actors
            if 'num_actors' not in input_dict:
                input_dict['num_actors'] = torch.tensor([len(idcs) for idcs in input_dict['actor_idcs']], 
                                                        device=input_dict['actor_idcs'][0].device)
        
        return batch_data 