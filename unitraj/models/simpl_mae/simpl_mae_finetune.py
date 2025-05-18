from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitraj.models.simpl_mae.simpl_mae_base import SimplMAEBase
from unitraj.models.simpl.simpl import MLPDecoder


class SimplMAEFinetune(SimplMAEBase):
    """
    SimplMAE fine-tuning model for trajectory prediction.
    Uses the pretrained encoder (FusionNet) and adds a multi-modal decoder for trajectory prediction.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # For fine-tuning, we use the MLPDecoder from SIMPL
        self.decoder = MLPDecoder(self.device, config)
        
        # Loss weights
        self.traj_loss_weight = config.get('traj_loss_weight', 1.0)
        self.mode_loss_weight = config.get('mode_loss_weight', 1.0)

        # Load pretrained model if specified
        if config.get('use_pretrained', False):
            self._load_pretrained_model(config)

    def forward(self, batch):
        """Forward pass for fine-tuning for trajectory prediction"""
        # Get pre-processed data
        input_dict = batch['input_dict']
        actors = input_dict['actors']
        actor_idcs = input_dict['actor_idcs']
        lanes = input_dict['lanes']
        lane_idcs = input_dict['lane_idcs']
        rpe_prep = input_dict['rpe_prep']
        actors_gt = input_dict['actors_gt']
        
        # Feature fusion using the pretrained encoder
        actors, lanes, _ = self.fusion_net(actors, actor_idcs, lanes, lane_idcs, rpe_prep)
        
        # Decode trajectories using the SIMPL decoder
        mode_logits, traj, vels = self.decoder(actors, actor_idcs)
          
        # Calculate loss
        loss_dict = self.compute_loss(mode_logits, traj, actors_gt, actor_idcs)
        
        # Convert loss values to float
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                loss_dict[k] = v.item()

        output = {}
        output['predicted_probability'] = mode_logits  
        output['predicted_trajectory'] = traj 
        
        return output, loss_dict['total_loss']

    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log_info(batch, batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='val')
        return loss

    def compute_loss(self, mode_logits, traj, actors_gt, actor_idcs):
        """
        Compute loss for trajectory prediction.
        
        Args:
            mode_logits: List of tensors containing mode probabilities for each batch
                        [batch_size] of [num_objects, num_modes]
            traj: List of tensors containing predicted trajectories for each batch
                 [batch_size] of [num_objects, num_modes, future_len, 2]
            actors_gt: Ground truth trajectories [total_num_objects, 4, future_len]
            actor_idcs: List of tensors containing indices for each batch
                       [batch_size] of [num_objects]
        """
        # Get loss coefficients from config
        cls_coef = self.config.get('cls_coef', 1.0)
        reg_coef = self.config.get('reg_coef', 1.0)
        loss_type = self.config.get('loss_type', 'ADE')  # 'ADE' or 'FDE'
        
        total_cls_loss = 0
        total_reg_loss = 0
        total_objects = 0

        # Process each batch
        for batch_idx, (batch_mode_logits, batch_traj, batch_idcs) in enumerate(zip(mode_logits, traj, actor_idcs)):
            # Get ground truth for this batch
            batch_gt = actors_gt[batch_idcs]
            batch_gt = batch_gt.permute(0, 2, 1)  
            num_objects = len(batch_idcs)
            total_objects += num_objects
            
            # Calculate trajectory loss (ADE or FDE)
            if loss_type == 'ADE':
                # Calculate ADE (Average Displacement Error)
                traj_diff = batch_traj - batch_gt[:, None, :, :2]  # [num_objects, num_modes, future_len, 2]
                traj_loss = torch.norm(traj_diff, dim=-1).mean(dim=-1)  # [num_objects, num_modes]
            else:  # FDE
                # Calculate FDE (Final Displacement Error)
                traj_diff = batch_traj[:, :, -1,:] - batch_gt[:,None, -1, :2]  # [num_objects, num_modes, 2]
                traj_loss = torch.norm(traj_diff, dim=-1)  # [num_objects, num_modes]
            
            # Get best mode for each object
            best_mode = traj_loss.argmin(dim=-1)  # [num_objects]
            
            # Calculate regression loss using best mode
            reg_loss = traj_loss[torch.arange(num_objects), best_mode].mean()
            total_reg_loss += reg_loss * num_objects
            
            # Calculate classification loss
            cls_loss = F.cross_entropy(batch_mode_logits, best_mode, reduction='mean')
            total_cls_loss += cls_loss * num_objects
        
        # Calculate average losses
        avg_cls_loss = total_cls_loss / total_objects
        avg_reg_loss = total_reg_loss / total_objects
        
        # Calculate total loss with coefficients
        total_loss = cls_coef * avg_cls_loss + reg_coef * avg_reg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': avg_cls_loss,
            'reg_loss': avg_reg_loss
        }

    def log_info(self, batch, batch_idx, prediction, status='train'):
        """Log information for training/validation"""
        # Log losses
        self.log(f"{status}/total_loss", prediction.get("total_loss", 0), on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{status}/cls_loss", prediction.get("cls_loss", 0), on_step=True, on_epoch=True)
        self.log(f"{status}/reg_loss", prediction.get("reg_loss", 0), on_step=True, on_epoch=True) 