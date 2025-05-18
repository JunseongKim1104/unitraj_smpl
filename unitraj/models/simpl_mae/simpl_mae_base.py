from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from math import gcd
import numpy as np
import math

from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.simpl.simpl import ActorNet, LaneNet, FusionNet, SftLayer


class SimplMAEBase(BaseModel):
    """
    Base class for SimplMAE models with common functionality.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Common parameters
        self.embed_dim = config.get('d_embed', 128)
        self.encoder_depth = config.get('n_scene_layer', 4)
        self.num_heads = config.get('n_scene_head', 8)
        self.drop_path = config.get('dropout', 0.2)
        self.history_steps = config.get('g_obs_len', 50)
        self.future_steps = config.get('g_pred_len', 60)
        
        # Create the fusion network (encoder)
        self.fusion_net = FusionNet(self.device, config)
    
    def _init_weights(self, m):
        """Initialize weights for linear and layer norm layers"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained_model(self, config):
        """Load pretrained model weights and handle finetuning if specified"""
        pretrained_path = config.get('pretrained_path')
        if pretrained_path is None:
            raise ValueError("pretrained_path must be specified when use_pretrained is True")
        
        # Load pretrained weights
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove module prefix if present (from DDP training)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load weights
        self.load_state_dict(state_dict, strict=False)
        
        # Handle layer finetuning
        finetune_layers = config.get('finetune_layers', [])
        
        if finetune_layers:
            # First freeze all layers
            for name, param in self.named_parameters():
                param.requires_grad = False
            
            # Then unfreeze specified layers
            for name, param in self.named_parameters():
                if any(layer in name for layer in finetune_layers):
                    param.requires_grad = True
                    print(f"Unfreezing layer for finetuning: {name}")
        else:
            # If no specific layers are specified, all layers are trainable
            print("No specific finetune layers specified. All layers will be trainable.")
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Get optimizer parameters from config
        opt_name = self.config.get('opt', 'adam')
        weight_decay = self.config.get('weight_decay', 0.0)
        init_lr = self.config.get('init_lr', 1e-4)
        
        # Create optimizer
        if opt_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=init_lr,
                weight_decay=weight_decay
            )
        elif opt_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=init_lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        # Get scheduler parameters
        scheduler_name = self.config.get('scheduler', 'polyline')
        milestones = self.config.get('milestones', [0, 5, 35, 40])
        values = self.config.get('values', [1e-4, 1e-3, 1e-3, 1e-4])
        
        # Create scheduler
        if scheduler_name == 'polyline':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda epoch: self._get_polyline_lr(epoch, milestones, values)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }
    
    def _get_polyline_lr(self, epoch, milestones, values):
        """Get learning rate based on polyline schedule"""
        for i in range(len(milestones) - 1):
            if milestones[i] <= epoch < milestones[i + 1]:
                return values[i]
        return values[-1] 