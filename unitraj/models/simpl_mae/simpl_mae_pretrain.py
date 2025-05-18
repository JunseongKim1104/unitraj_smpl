from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from unitraj.models.simpl_mae.simpl_mae_base import SimplMAEBase
from unitraj.models.simpl.simpl import SftLayer


class SimplMAEPretrain(SimplMAEBase):
    """
    SimplMAE pretraining model with masked autoencoder approach.
    Predicts masked history, future trajectories, and lane segments.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # MAE specific parameters
        self.actor_mask_ratio = config.get('actor_mask_ratio', 0.5)
        self.lane_mask_ratio = config.get('lane_mask_ratio', 0.5)
        self.loss_weight = config.get('mae_loss_weight', [1.0, 1.0, 0.35])
        self.decoder_depth = config.get('decoder_depth', 4)
        self.mlp_ratio = config.get('mlp_ratio', 4.0)
        self.qkv_bias = config.get('qkv_bias', False)
        
        # For pretraining, we need the MAE decoder
        self.decoder_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Decoder transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path, self.decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            SftLayer(
                device=self.device,
                d_edge=config.get('d_rpe', 128),
                d_model=self.embed_dim,
                d_ffn=self.embed_dim * self.mlp_ratio,
                n_head=self.num_heads,
                dropout=self.drop_path,
                update_edge=config.get('update_edge', True)
            )
            for i in range(self.decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(self.embed_dim)

        # Mask tokens
        self.lane_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.future_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.history_mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Prediction heads
        self.future_pred = nn.Linear(self.embed_dim, self.future_steps * 2)
        self.history_pred = nn.Linear(self.embed_dim, self.history_steps * 2)
        self.lane_pred = nn.Linear(self.embed_dim, 20 * 2)  # Assuming 20 points per lane segment
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the MAE components"""
        nn.init.normal_(self.lane_mask_token, std=0.02)
        nn.init.normal_(self.future_mask_token, std=0.02)
        nn.init.normal_(self.history_mask_token, std=0.02)

        self.apply(self._init_weights)

    @staticmethod
    def agent_random_masking(hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors):
        """
        Randomly mask actor tokens, ensuring each actor has either history or future visible, not both
        """
        pred_masks = ~future_padding_mask.all(-1)  # [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int()
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []

        device = hist_tokens.device
        agent_ids = torch.arange(hist_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):
            pred_agent_ids = agent_ids[future_pred_mask]
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep]
            fut_ids_keep = pred_agent_ids[fut_ids_keep]
            fut_keep_ids_list.append(fut_ids_keep)

            hist_keep_mask = torch.zeros_like(agent_ids).bool()
            hist_keep_mask[: num_actors[i]] = True
            hist_keep_mask[fut_ids_keep] = False
            hist_ids_keep = agent_ids[hist_keep_mask]
            hist_keep_ids_list.append(hist_ids_keep)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])
            hist_masked_tokens.append(hist_tokens[i, hist_ids_keep])

            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))
            hist_key_padding_mask.append(torch.zeros(len(hist_ids_keep), device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )

    @staticmethod
    def lane_random_masking(x, lane_mask_ratio, key_padding_mask):
        """Randomly mask lane tokens"""
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - lane_mask_ratio)).int()

        x_masked, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)
            x_masked.append(x[i, ids_keep])
            new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))

        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list

    def forward(self, batch):
        """Forward pass for pretraining with masked autoencoder"""
        # Get pre-processed data
        input_dict = batch['input_dict']
        
        # Extract data from input_dict
        actors = input_dict['actors']
        actor_idcs = input_dict['actor_idcs']
        lanes = input_dict['lanes']
        lane_idcs = input_dict['lane_idcs']
        rpe_prep = input_dict['rpe_prep']
        actors_gt = input_dict['actors_gt']
        
        # Extract history and future data
        hist_padding_mask = input_dict['x_padding_mask'][:, :, :self.history_steps]
        future_padding_mask = input_dict['x_padding_mask'][:, :, self.history_steps:]
        
        # Get actor and lane features using FusionNet components
        actor_feats = self.fusion_net.actor_net(actors)
        lane_feats = self.fusion_net.lane_net(lanes)
        
        # Project features
        actor_feats = self.fusion_net.proj_actor(actor_feats)
        lane_feats = self.fusion_net.proj_lane(lane_feats)
        
        # Split actor features into history and future
        B, N = actor_feats.shape[0], actor_feats.shape[1]
        hist_feat = actor_feats
        future_feat = actor_feats  # In pretraining, we use the same features for future prediction
        
        # Get position embeddings
        x_centers = torch.cat(
            [input_dict['x_centers'], input_dict['x_centers'], input_dict['lane_centers']], dim=1
        )
        angles = torch.cat(
            [
                input_dict['x_angles'][..., self.history_steps-1],
                input_dict['x_angles'][..., self.history_steps-1],
                input_dict['lane_angles'],
            ],
            dim=1,
        )
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        
        # Apply masking to actor history, future, and lanes
        (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            input_dict['num_actors'],
        )
        
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_feats, self.lane_mask_ratio, input_dict['lane_key_padding_mask']
        )
        
        # Concatenate masked tokens for encoder
        x = torch.cat(
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )
        
        # Process through SFT encoder
        for i, block in enumerate(self.fusion_net.sft.layers):
            x = block(x, None, key_padding_mask)
        
        # Decode
        x_decoder = self.decoder_embed(x)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1],
            fut_masked_tokens.shape[1],
            lane_masked_tokens.shape[1],
        )
        
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh : Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]
        
        # Reconstruct tokens with mask tokens
        decoder_hist_token = self.history_mask_token.repeat(B, N, 1)
        hist_pred_mask = ~input_dict['x_key_padding_mask']
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)]
            hist_pred_mask[i, idx] = False
        
        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~input_dict['x_key_padding_mask']
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False
        
        decoder_lane_token = self.lane_mask_token.repeat(B, lane_feats.shape[1], 1)
        lane_pred_mask = ~input_dict['lane_key_padding_mask']
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False
        
        # Concatenate decoder tokens
        x_decoder = torch.cat(
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )
        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat)
        
        # Create decoder padding mask
        decoder_key_padding_mask = torch.cat(
            [
                input_dict['x_key_padding_mask'],
                future_padding_mask.all(-1),
                input_dict['lane_key_padding_mask'],
            ],
            dim=1,
        )
        
        # Process through decoder blocks
        for block in self.decoder_blocks:
            x_decoder = block(x_decoder, None, decoder_key_padding_mask)
        
        x_decoder = self.decoder_norm(x_decoder)
        
        # Get token outputs
        hist_token = x_decoder[:, :N]
        future_token = x_decoder[:, N : 2 * N]
        lane_token = x_decoder[:, -lane_feats.shape[1]:]
        
        # Lane prediction loss
        lane_positions = input_dict['lane_positions'] - input_dict['lane_centers'].unsqueeze(-2)
        lane_pred = self.lane_pred(lane_token).view(B, lane_feats.shape[1], 20, 2)
        lane_reg_mask = ~input_dict['lane_padding_mask']
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_positions[lane_reg_mask]
        )
        
        # History prediction loss
        x_hat = self.history_pred(hist_token).view(B, N, self.history_steps, 2)
        x = (input_dict['x_positions'] - input_dict['x_centers'].unsqueeze(-2))
        x_reg_mask = ~hist_padding_mask
        x_reg_mask[~hist_pred_mask] = False
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x[x_reg_mask])
        
        # Future prediction loss
        y_hat = self.future_pred(future_token).view(B, N, self.future_steps, 2)
        y = input_dict['y']
        reg_mask = ~future_padding_mask
        reg_mask[~future_pred_mask] = False
        future_loss = F.l1_loss(y_hat[reg_mask], y[reg_mask])
        
        # Total loss
        loss = (
            self.loss_weight[0] * future_loss
            + self.loss_weight[1] * hist_loss
            + self.loss_weight[2] * lane_pred_loss
        )
        
        loss_dict = {
            "total_loss": loss,
            "hist_loss": hist_loss.item(),
            "future_loss": future_loss.item(),
            "lane_pred_loss": lane_pred_loss.item(),
        }
        
        output = {
            "predicted_trajectory": y_hat,
            "predicted_history": x_hat,
            "predicted_lane": lane_pred
        }
        
        return output, loss_dict["total_loss"]

    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log_info(batch, batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log_info(batch, batch_idx, prediction, status='val')
        return loss

    def log_info(self, batch, batch_idx, prediction, status='train'):
        """Log information for training/validation"""
        self.log(f"{status}/total_loss", prediction.get("total_loss", 0), on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{status}/hist_loss", prediction.get("hist_loss", 0), on_step=True, on_epoch=True)
        self.log(f"{status}/future_loss", prediction.get("future_loss", 0), on_step=True, on_epoch=True)
        self.log(f"{status}/lane_pred_loss", prediction.get("lane_pred_loss", 0), on_step=True, on_epoch=True) 