from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
# from fractions import gcd
from math import gcd
import numpy as np
import math
#

from unitraj.models.base_model.base_model import BaseModel


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(
            int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, n_in=14, hidden_size=128, n_fpn_scale=4):
        super(ActorNet, self).__init__()
        norm = "GN"
        ng = 1

        # FPN scales: [32, 64, 128, 256]
        n_out = [2**(5 + s) for s in range(n_fpn_scale)]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale

        # Feature extraction groups
        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        # Lateral connections
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        # Output layer
        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        # Feature extraction
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        # Feature fusion with lateral connections
        out = self.lateral[-1](outputs[-1])
        
        for i in range(len(outputs) - 2, -1, -1):
            # Get target size from lateral input
            target_size = outputs[i].shape[-1]
            # Resize to match lateral input size
            out = F.interpolate(out, size=target_size, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        # Final output
        out = self.output(out)[:, :, -1]
        return out


class PointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointAggregateBlock, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _global_maxpool_aggre(self, feat):
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x_inp):
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]
        x_aggre = self._global_maxpool_aggre(x)
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)

        out = self.norm(x_inp + self.fc2(x_aggre))
        if self.aggre_out:
            return self._global_maxpool_aggre(out).squeeze()
        else:
            return out


class LaneNet(nn.Module):
    def __init__(self, device, in_size=24, hidden_size=128, dropout=0.1):
        super(LaneNet, self).__init__()
        self.device = device

        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )

        # Feature aggregation blocks
        self.aggre1 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=False, dropout=dropout)
        self.aggre2 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=True, dropout=dropout)

    def forward(self, feats):
        # Initial projection
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size]
        
        # First aggregation (keep sequence)
        x = self.aggre1(x)  # [N_{lane}, 10, hidden_size]
        
        # Second aggregation (global)
        x = self.aggre2(x)  # [N_{lane}, hidden_size]
        
        return x


class SftLayer(nn.Module):
    def __init__(self,
                 device,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 2048,
                 n_head: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True) -> None:
        super(SftLayer, self).__init__()
        self.device = device
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                node: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                node:       (N, d_model)
                edge:       (N, N, d_model)
                edge_mask:  (N, N)
        '''
        # update node
        x, edge, memory = self._build_memory(node, edge)
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))
        return x, edge, None

    def _build_memory(self,
                      node: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            input:
                node:   (N, d_model)
                edge:   (N, N, d_edge)
            output:
                :param  (1, N, d_model)
                :param  (N, N, d_edge)
                :param  (N, N, d_model)
        '''
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    def _mha_block(self,
                   x: Tensor,
                   mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  [1, N, d_model]
                mem:                [N, N, d_model]
                attn_mask:          [N, N]
                key_padding_mask:   [N, N]
            output:
                :param      [1, N, d_model]
                :param      [N, N]
        '''
        x, _ = self.multihead_attn(x, mem, mem,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(self,
                 device,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 6,
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(SymmetricFusionTransformer, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(SftLayer(device=device,
                                   d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model*2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        '''
            x: (N, d_model)
            edge: (d_model, N, N)
            edge_mask: (N, N)
        '''
        # attn_multilayer = []
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask)
            # attn_multilayer.append(attn)
        return x, None


class FusionNet(nn.Module):
    def __init__(self, device, config):
        super(FusionNet, self).__init__()
        self.device = device
        self.config = config
        
        # Projection layers
        self.proj_actor = nn.Sequential(
            nn.Linear(config.get('d_actor', 128), config.get('d_embed', 128)),
            nn.LayerNorm(config.get('d_embed', 128)),
            nn.ReLU(inplace=True)
        )
        
        self.proj_lane = nn.Sequential(
            nn.Linear(config.get('d_lane', 128), config.get('d_embed', 128)),
            nn.LayerNorm(config.get('d_embed', 128)),
            nn.ReLU(inplace=True)
        )
        
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(config.get('d_rpe_in', 5), config.get('d_rpe', 128)),
            nn.LayerNorm(config.get('d_rpe', 128)),
            nn.ReLU(inplace=True)
        )

        self.actor_net = ActorNet(
            n_in=config.get('in_actor', 14),
            hidden_size=config.get('d_embed', 128),
            n_fpn_scale=config.get('n_fpn_scale', 4)
        )

        self.lane_net = LaneNet(
            device=device,
            in_size=config.get('in_lane', 24),
            hidden_size=config.get('d_embed', 128),
            dropout=config.get('dropout', 0.1)
        )

        self.sft = SymmetricFusionTransformer(
            device=device,
            d_model=config.get('d_embed', 128),
            d_edge=config.get('d_rpe', 128),
            n_head=config.get('n_scene_head', 8),
            n_layer=config.get('n_scene_layer', 4),
            dropout=config.get('dropout', 0.1),
            update_edge=config.get('update_edge', True)
        )

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor]):
        # Actor feature extraction
        actor_feats = self.actor_net(actors)  # [batch_size, num_valid_objects, seq_len, d_embed]
        
        # Lane feature extraction
        lane_feats = self.lane_net(lanes)  # [batch_size, num_valid_polylines, num_points, d_embed]
        
        # Project features
        actor_feats = self.proj_actor(actor_feats)
        lane_feats = self.proj_lane(lane_feats)
        
        # Process each batch separately
        actors_new, lanes_new = [], []
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep):
            # Get features for this batch
            _actors = actor_feats[a_idcs]  # [num_valid_objects, seq_len, d_embed]
            _lanes = lane_feats[l_idcs]  # [num_valid_polylines, num_points, d_embed]
            
            # Concatenate features
            tokens = torch.cat([_actors, _lanes], dim=0)  # [num_total, seq_len, d_embed]
            
            # Process RPE
            rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))  # [num_total, num_total, d_rpe]
            
            # Fusion
            out, _ = self.sft(tokens, rpe, rpes['scene_mask'])
            
            # Split results
            actors_new.append(out[:len(a_idcs)])
            lanes_new.append(out[len(a_idcs):])
        
        # Concatenate results
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        
        return actors, lanes, None


class MLPDecoder(nn.Module):
    def __init__(self,
                 device,
                 config) -> None:
        super(MLPDecoder, self).__init__()
        self.device = device
        self.config = config
        self.hidden_size = config.get('d_embed', 128)
        self.future_steps = config.get('g_pred_len', 60)
        self.num_modes = config.get('g_num_modes', 6)
        self.param_out = config.get('param_out', 'bezier')  # parametric output: bezier/monomial/none
        self.N_ORDER = config.get('param_order', 7)

        dim_mm = self.hidden_size * self.num_modes
        dim_inter = dim_mm // 2
        self.multihead_proj = nn.Sequential(
            nn.Linear(self.hidden_size, dim_inter),
            nn.LayerNorm(dim_inter),
            nn.ReLU(inplace=True),
            nn.Linear(dim_inter, dim_mm),
            nn.LayerNorm(dim_mm),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)
        )

        if self.param_out == 'bezier':
            self.mat_T = self._get_T_matrix_bezier(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_bezier(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2)
            )
        elif self.param_out == 'monomial':
            self.mat_T = self._get_T_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_monomial(n_order=self.N_ORDER, n_step=self.future_steps).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2)
            )
        elif self.param_out == 'none':
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2)
            )
        else:
            raise NotImplementedError

    def _get_T_matrix_bezier(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = math.comb(n_order, i) * (1.0 - ts)**(n_order - i) * ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_bezier(self, n_order, n_step):
        # ~ 1st derivatives
        # ! NOTICE: we multiply n_order inside of the Tp matrix
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = n_order * math.comb(n_order - 1, i) * (1.0 - ts)**(n_order - 1 - i) * ts**i
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def _get_T_matrix_monomial(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = ts ** i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_monomial(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (i + 1) * (ts ** i)
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def forward(self,
                embed: torch.Tensor,
                actor_idcs: List[Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # input embed: [159, 128]
        embed = self.multihead_proj(embed).view(-1, self.num_modes, self.hidden_size).permute(1, 0, 2)
        # print('embed: ', embed.shape)  # e.g., [6, 159, 128]

        cls = self.cls(embed).view(self.num_modes, -1).permute(1, 0)  # e.g., [159, 6]
        cls = F.softmax(cls * 1.0, dim=1)  # e.g., [159, 6]

        if self.param_out == 'bezier':
            param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 2)  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, torch.diff(param, dim=2)) / (self.future_steps * 0.1)
        elif self.param_out == 'monomial':
            param = self.reg(embed).view(self.num_modes, -1, self.N_ORDER + 1, 2)  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, param[:, :, 1:, :]) / (self.future_steps * 0.1)
        elif self.param_out == 'none':
            reg = self.reg(embed).view(self.num_modes, -1, self.future_steps, 2)  # e.g., [6, 159, 30, 2]
            reg = reg.permute(1, 0, 2, 3)  # e.g., [159, 6, 30, 2]
            vel = torch.gradient(reg, dim=-2)[0] / 0.1  # vel is calculated from pos

        # print('reg: ', reg.shape, 'cls: ', cls.shape)
        # de-batchify
        res_cls, res_reg, res_aux = [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            res_cls.append(cls[idcs])
            res_reg.append(reg[idcs])

            if self.param_out == 'none':
                res_aux.append((vel[idcs], None))  # ! None is a placeholder
            else:
                res_aux.append((vel[idcs], param[idcs]))  # List[Tuple[Tensor,...]]

        return res_cls, res_reg, res_aux


class Simpl(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.fusion_net = FusionNet(self.device, config)
        self.decoder = MLPDecoder(self.device, config)
        
        # Loss weights
        self.traj_loss_weight = config.get('traj_loss_weight', 1.0)
        self.mode_loss_weight = config.get('mode_loss_weight', 1.0)

        # Load pretrained model if specified
        if config.get('use_pretrained', False):
            self._load_pretrained_model(config)

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

    def forward(self, batch):
        # Get pre-processed data
        input_dict = batch['input_dict']
        actors = input_dict['actors']
        actor_idcs = input_dict['actor_idcs']
        lanes = input_dict['lanes']
        lane_idcs = input_dict['lane_idcs']
        rpe_prep = input_dict['rpe_prep']
        actors_gt = input_dict['actors_gt']  # Add actors_gt
        
        # Feature fusion
        actors, lanes, _ = self.fusion_net(actors, actor_idcs, lanes, lane_idcs, rpe_prep)
        
        # Decode trajectories
        mode_logits, traj, vels = self.decoder(actors, actor_idcs)
          
        # Calculate loss
        loss = self.compute_loss(mode_logits, traj, actors_gt, actor_idcs)
        

        output = {}
        output['predicted_probability'] = mode_logits  
        output['predicted_trajectory'] = traj 
        
        return output, loss['total_loss']

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

    def pre_process(self, batch):
        """
        Preprocess the input batch data for SIMPL model.
        
        Args:
            batch: Dictionary containing input data with the following keys:
                - obj_trajs: Historical trajectories of objects [N, T, 39]
                - obj_trajs_mask: Valid mask for obj_trajs [N, T]
                - track_index_to_predict: Index of trajectory to predict
                - obj_trajs_pos: Position coordinates [N, T, 3]
                - obj_trajs_last_pos: Last position coordinates [N, 3]
                - center_objects_world: World coordinates of centered objects
                - center_objects_id: ID of centered objects
                - center_objects_type: Type of centered objects
                - map_center: World coordinates of map center
                - map_polylines: Map polylines [M, P, 29]
                - map_polylines_mask: Valid mask for map_polylines [M, P]
                - map_polylines_center: Center points of map polylines [M, 3]
                - obj_trajs_future_state: Future state of objects
                - obj_trajs_future_mask: Valid mask for future states
                - center_gt_trajs: Ground truth trajectories
                - center_gt_trajs_mask: Valid mask for ground truth
                - center_gt_final_valid_idx: Final valid index
                - center_gt_trajs_src: Ground truth in world coordinates
                - dataset_name: Name of dataset
                - kalman_difficulty: Kalman filter difficulty
                - trajectory_type: Type of trajectory
                
        Returns:
            Tuple containing:
            - actors: Tensor of actor features [total_num_valid_objects, feature_dim, seq_len]
            - actor_idcs: List of actor indices for each batch
            - lanes: Tensor of lane features [total_num_valid_lanes, num_points, feature_dim]
            - lane_idcs: List of lane indices for each batch
            - rpe_prep: List of dictionaries containing relative position encoding data
        """
        input_dict = batch['input_dict']
        
        # Extract basic information
        obj_trajs = input_dict['obj_trajs']  # [N, T, 39]
        obj_trajs_mask = input_dict['obj_trajs_mask']  # [N, T]
        track_index_to_predict = input_dict['track_index_to_predict']
        
        # Get center object information
        center_objects_world = input_dict['center_objects_world']
        center_objects_type = input_dict['center_objects_type']
        
        # Extract map information
        map_polylines = input_dict['map_polylines']  # [M, P, 29]
        map_polylines_mask = input_dict['map_polylines_mask']  # [M, P]
        
        batch_size = obj_trajs.shape[0]
        num_objects = obj_trajs.shape[1]
        num_polylines = map_polylines.shape[1]
        
        # Process actors data
        actors = []
        actor_idcs = []
        total_actors = 0
        
        for b in range(batch_size):
            # Get valid objects in this batch
            valid_mask = obj_trajs_mask[b].any(dim=-1)  # [num_objects]
            valid_objects = obj_trajs[b, valid_mask]  # [num_valid_objects, seq_len, 39]
            
            # Extract position and heading
            trajs_pos = valid_objects[..., :3]  # [num_valid_objects, seq_len, 3]
            trajs_heading = valid_objects[..., 33:35]  # [num_valid_objects, seq_len, 2]
            
            # Get last time point position and heading for each object
            last_pos = trajs_pos[:, -1]  # [num_valid_objects, 3]
            last_heading = trajs_heading[:, -1]  # [num_valid_objects, 2]
            
            # Calculate rotation angles for each object
            thetas = torch.atan2(last_heading[:, 1], last_heading[:, 0])  # [num_valid_objects]
            
            # Create rotation matrices for each object
            cos_thetas = torch.cos(thetas)  # [num_valid_objects]
            sin_thetas = torch.sin(thetas)  # [num_valid_objects]
            
            # Create rotation matrices [num_valid_objects, 2, 2]
            rot_matrices = torch.stack([
                torch.stack([cos_thetas, -sin_thetas], dim=1),
                torch.stack([sin_thetas, cos_thetas], dim=1)
            ], dim=2)
            
            # First translate to origin (subtract last position)
            trajs_pos_centered = trajs_pos - last_pos.unsqueeze(1)  # [num_valid_objects, seq_len, 3]
            
            # Then rotate around origin
            # Reshape for batch matrix multiplication
            pos_reshaped = trajs_pos_centered[..., :2]  # [num_valid_objects, seq_len, 2]
            rot_reshaped = rot_matrices.unsqueeze(1)  # [num_valid_objects, 1, 2, 2]
            
            # Apply rotation to all trajectories at once
            pos_rotated = torch.matmul(pos_reshaped.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [num_valid_objects, seq_len, 2]
            trajs_pos_centered[..., :2] = pos_rotated
            
            # Transform velocities
            trajs_vel = valid_objects[..., 35:37]  # [num_valid_objects, seq_len, 2]
            vel_rotated = torch.matmul(trajs_vel.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [num_valid_objects, seq_len, 2]
            trajs_vel = vel_rotated
            
            # Transform headings
            # For headings, we need to rotate the heading vectors
            heading_rotated = torch.matmul(trajs_heading.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [num_valid_objects, seq_len, 2]
            trajs_heading = heading_rotated
            
            # Extract object type
            trajs_type = valid_objects[..., 6:13]  # [num_valid_objects, seq_len, 7]
            
            # Get padding mask
            trajs_mask = obj_trajs_mask[b, valid_mask]  # [num_valid_objects, seq_len]
            
            # Combine features
            actor_features = torch.cat([
                trajs_pos_centered[..., :2],  # position (x, y)
                trajs_heading,  # heading encoding (sin, cos)
                trajs_vel,  # velocity (vx, vy)
                trajs_type,  # object type one-hot
                trajs_mask.unsqueeze(-1),  # padding mask
            ], dim=-1)  # [num_valid_objects, seq_len, feature_dim]
            
            # Exclude the last frame
            # actor_features = actor_features[:, :-1, :]  # [num_valid_objects, 20, feature_dim]
            
            actors.append(actor_features)
            
            # Create actor indices
            actor_idx = torch.arange(total_actors, total_actors + len(valid_objects), device=self.device)
            actor_idcs.append(actor_idx)
            total_actors += len(valid_objects)
        
        # Process map data
        lanes = []
        lane_idcs = []
        total_lanes = 0
        
        for b in range(batch_size):
            # Get valid polylines in this batch
            valid_mask = map_polylines_mask[b].any(dim=-1)  # [num_polylines]
            valid_polylines = map_polylines[b, valid_mask]  # [num_valid_polylines, num_points, 29]
            
            # Extract position and direction
            lane_pos = valid_polylines[..., :3]  # [num_valid_polylines, num_points, 3]
            lane_dir = valid_polylines[..., 3:6]  # [num_valid_polylines, num_points, 3]
            
            # Calculate center point and heading for each lane
            lane_centers = lane_pos.mean(dim=1)  # [num_valid_polylines, 3]
            lane_vectors = lane_pos[:, -1] - lane_pos[:, 0]  # [num_valid_polylines, 3]
            lane_headings = torch.atan2(lane_vectors[:, 1], lane_vectors[:, 0])  # [num_valid_polylines]
            
            # Calculate rotation angles for each lane
            cos_thetas = torch.cos(lane_headings)  # [num_valid_polylines]
            sin_thetas = torch.sin(lane_headings)  # [num_valid_polylines]
            
            # Create rotation matrices [num_valid_polylines, 2, 2]
            rot_matrices = torch.stack([
                torch.stack([cos_thetas, -sin_thetas], dim=1),
                torch.stack([sin_thetas, cos_thetas], dim=1)
            ], dim=2)
            
            # First translate to center (subtract lane center)
            lane_pos_centered = lane_pos - lane_centers.unsqueeze(1)  # [num_valid_polylines, num_points, 3]
            
            # Then rotate around center
            # Reshape for batch matrix multiplication
            pos_reshaped = lane_pos_centered[..., :2]  # [num_valid_polylines, num_points, 2]
            rot_reshaped = rot_matrices.unsqueeze(1)  # [num_valid_polylines, 1, 2, 2]
            
            # Apply rotation to all lanes at once
            pos_rotated = torch.matmul(pos_reshaped.unsqueeze(-2), rot_reshaped).squeeze(-2)  # [num_valid_polylines, num_points, 2]
            lane_pos_centered[..., :2] = pos_rotated
            
            # Transform directions
            dir_rotated = torch.matmul(lane_dir[..., :2].unsqueeze(-2), rot_reshaped).squeeze(-2)  # [num_valid_polylines, num_points, 2]
            lane_dir[..., :2] = dir_rotated
            
            # Extract lane type
            lane_type = valid_polylines[..., 9:29]  # [num_valid_polylines, num_points, 20]
            
            # Combine features
            lane_features = torch.cat([
                lane_pos_centered[..., :2],  # position (x, y)
                lane_dir[..., :2],  # direction (x, y)
                lane_type,  # lane type one-hot
            ], dim=-1)  # [num_valid_polylines, num_points, feature_dim]
            
            lanes.append(lane_features)
            
            # Create lane indices
            lane_idx = torch.arange(total_lanes, total_lanes + len(valid_polylines), device=self.device)
            lane_idcs.append(lane_idx)
            total_lanes += len(valid_polylines)
        
        # Prepare relative position encoding
        rpe_prep = []
        for b in range(batch_size):
            # Get original actor and lane features before localization
            valid_mask = obj_trajs_mask[b].any(dim=-1)  # [num_objects]
            valid_objects = obj_trajs[b, valid_mask]  # [num_valid_objects, seq_len, 39]
            
            # Get original actor positions and headings
            actor_pos = valid_objects[..., :3]  # [num_valid_objects, seq_len, 3]
            actor_heading = valid_objects[..., 33:35]  # [num_valid_objects, seq_len, 2]
            
            # Get original lane positions and directions
            valid_lane_mask = map_polylines_mask[b].any(dim=-1)  # [num_polylines]
            valid_polylines = map_polylines[b, valid_lane_mask]  # [num_valid_polylines, num_points, 29]
            lane_pos = valid_polylines[..., :3]  # [num_valid_polylines, num_points, 3]
            lane_dir = valid_polylines[..., 3:6]  # [num_valid_polylines, num_points, 3]
            
            # Calculate relative positions and headings
            num_actors = actor_pos.shape[0]
            num_lanes = lane_pos.shape[0]
            total_nodes = num_actors + num_lanes
            
            # Initialize RPE tensors
            scene_rpe = torch.zeros(5, total_nodes, total_nodes, device=self.device)
            scene_mask = torch.zeros(total_nodes, total_nodes, device=self.device)
            
            # Calculate actor-actor RPE
            for i in range(num_actors):
                for j in range(num_actors):
                    if i != j:
                        # Get last time point positions and headings
                        pos_i = actor_pos[i, -1]  # [3]
                        pos_j = actor_pos[j, -1]  # [3]
                        heading_i = actor_heading[i, -1]  # [2]
                        heading_j = actor_heading[j, -1]  # [2]
                        
                        # Relative position
                        rel_pos = pos_i[:2] - pos_j[:2]  # [2]
                        rel_dist = torch.norm(rel_pos)
                        
                        # Relative heading
                        heading_i_angle = torch.atan2(heading_i[1], heading_i[0])
                        heading_j_angle = torch.atan2(heading_j[1], heading_j[0])
                        rel_heading = heading_i_angle - heading_j_angle
                        rel_heading_sin = torch.sin(rel_heading)
                        rel_heading_cos = torch.cos(rel_heading)
                        
                        # Angle between heading and position vector
                        pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
                        angle_diff = pos_angle - heading_i_angle
                        angle_diff_sin = torch.sin(angle_diff)
                        angle_diff_cos = torch.cos(angle_diff)
                        
                        # Fill RPE
                        scene_rpe[0, i, j] = rel_heading_sin
                        scene_rpe[1, i, j] = rel_heading_cos
                        scene_rpe[2, i, j] = angle_diff_sin
                        scene_rpe[3, i, j] = angle_diff_cos
                        scene_rpe[4, i, j] = rel_dist
                        scene_mask[i, j] = 1
            
            # Calculate actor-lane RPE
            for i in range(num_actors):
                for j in range(num_lanes):
                    # Get actor last time point position and heading
                    pos_i = actor_pos[i, -1]  # [3]
                    heading_i = actor_heading[i, -1]  # [2]
                    
                    # Get lane center and heading
                    lane_center = lane_pos[j].mean(dim=0)  # [3]
                    lane_vector = lane_pos[j, -1] - lane_pos[j, 0]  # [3]
                    lane_heading = torch.atan2(lane_vector[1], lane_vector[0])
                    
                    # Relative position
                    rel_pos = pos_i[:2] - lane_center[:2]  # [2]
                    rel_dist = torch.norm(rel_pos)
                    
                    # Relative heading
                    heading_i_angle = torch.atan2(heading_i[1], heading_i[0])
                    rel_heading = heading_i_angle - lane_heading
                    rel_heading_sin = torch.sin(rel_heading)
                    rel_heading_cos = torch.cos(rel_heading)
                    
                    # Angle between heading and position vector
                    pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
                    angle_diff = pos_angle - heading_i_angle
                    angle_diff_sin = torch.sin(angle_diff)
                    angle_diff_cos = torch.cos(angle_diff)
                    
                    # Fill RPE
                    scene_rpe[0, i, num_actors + j] = rel_heading_sin
                    scene_rpe[1, i, num_actors + j] = rel_heading_cos
                    scene_rpe[2, i, num_actors + j] = angle_diff_sin
                    scene_rpe[3, i, num_actors + j] = angle_diff_cos
                    scene_rpe[4, i, num_actors + j] = rel_dist
                    scene_mask[i, num_actors + j] = 1
            
            # Calculate lane-lane RPE
            for i in range(num_lanes):
                for j in range(num_lanes):
                    if i != j:
                        # Get lane centers and headings
                        lane_center_i = lane_pos[i].mean(dim=0)  # [3]
                        lane_center_j = lane_pos[j].mean(dim=0)  # [3]
                        lane_vector_i = lane_pos[i, -1] - lane_pos[i, 0]  # [3]
                        lane_vector_j = lane_pos[j, -1] - lane_pos[j, 0]  # [3]
                        lane_heading_i = torch.atan2(lane_vector_i[1], lane_vector_i[0])
                        lane_heading_j = torch.atan2(lane_vector_j[1], lane_vector_j[0])
                        
                        # Relative position
                        rel_pos = lane_center_i[:2] - lane_center_j[:2]  # [2]
                        rel_dist = torch.norm(rel_pos)
                        
                        # Relative heading
                        rel_heading = lane_heading_i - lane_heading_j
                        rel_heading_sin = torch.sin(rel_heading)
                        rel_heading_cos = torch.cos(rel_heading)
                        
                        # Angle between heading and position vector
                        pos_angle = torch.atan2(rel_pos[1], rel_pos[0])
                        angle_diff = pos_angle - lane_heading_i
                        angle_diff_sin = torch.sin(angle_diff)
                        angle_diff_cos = torch.cos(angle_diff)
                        
                        # Fill RPE
                        scene_rpe[0, num_actors + i, num_actors + j] = rel_heading_sin
                        scene_rpe[1, num_actors + i, num_actors + j] = rel_heading_cos
                        scene_rpe[2, num_actors + i, num_actors + j] = angle_diff_sin
                        scene_rpe[3, num_actors + i, num_actors + j] = angle_diff_cos
                        scene_rpe[4, num_actors + i, num_actors + j] = rel_dist
                        scene_mask[num_actors + i, num_actors + j] = 1
            
            rpe_prep.append({
                'scene': scene_rpe,
                'scene_mask': scene_mask
            })
        
        # Convert lists to tensors
        actors = torch.cat(actors, dim=0)  # [total_num_valid_objects, seq_len, feature_dim]
        actors = actors.permute(0, 2, 1)  # [total_num_valid_objects, feature_dim, seq_len]
        lanes = torch.cat(lanes, dim=0)  # [total_num_valid_lanes, num_points, feature_dim]
        
        return actors, actor_idcs, lanes, lane_idcs, rpe_prep

    def post_process(self, mode_logits, traj):
        return {
            'predicted_probability': F.softmax(mode_logits, dim=-1),
            'predicted_trajectory': traj
        } 

    def log_info(self, batch, batch_idx, prediction, status='train'):
        """
        Log information for training/validation.
        
        Args:
            batch: Dictionary containing input data
            batch_idx: Current batch index
            prediction: Dictionary containing model predictions
            status: 'train' or 'val'
        """
        # Get predictions
        mode_logits = prediction['predicted_probability']  # List of [num_objects, num_modes]
        traj = prediction['predicted_trajectory']  # List of [num_objects, num_modes, future_len, 2]
        
        # Get ground truth and indices
        actors_gt = batch['input_dict']['actors_gt']  # [total_num_objects, 4, future_len]
        actor_idcs = batch['input_dict']['actor_idcs']  # List of [num_objects]
        
        # Get loss coefficients and type from config
        cls_coef = self.config.get('cls_coef', 1.0)
        reg_coef = self.config.get('reg_coef', 1.0)
        loss_type = self.config.get('loss_type', 'ADE')  # 'ADE' or 'FDE'
        
        # Initialize metrics
        total_cls_loss = 0
        total_reg_loss = 0
        total_objects = 0
        
        # Initialize lists to store metrics for overall calculation
        all_min_ade = []
        all_min_fde = []
        all_miss_rate = []
        all_brier_fde = []
        
        # Process each batch
        for batch_idx, (batch_mode_logits, batch_traj, batch_idcs) in enumerate(zip(mode_logits, traj, actor_idcs)):
            # Get ground truth for this batch
            batch_gt = actors_gt[batch_idcs]
            batch_gt = batch_gt.permute(0, 2, 1)  # [num_objects, future_len, 4]
            num_objects = len(batch_idcs)
            total_objects += num_objects
            
            # Calculate trajectory loss (ADE or FDE)
            if loss_type == 'ADE':
                # Calculate ADE (Average Displacement Error)
                traj_diff = batch_traj - batch_gt[:, None, :, :2]  # [num_objects, num_modes, future_len, 2]
                traj_loss = torch.norm(traj_diff, dim=-1).mean(dim=-1)  # [num_objects, num_modes]
            else:  # FDE
                # Calculate FDE (Final Displacement Error)
                traj_diff = batch_traj[:, :, -1, :] - batch_gt[:, None, -1, :2]  # [num_objects, num_modes, 2]
                traj_loss = torch.norm(traj_diff, dim=-1)  # [num_objects, num_modes]
            
            # Get best mode for each object
            best_mode = traj_loss.argmin(dim=-1)  # [num_objects]
            
            # Calculate regression loss using best mode
            reg_loss = traj_loss[torch.arange(num_objects), best_mode].mean()
            total_reg_loss += reg_loss * num_objects
            
            # Calculate classification loss
            cls_loss = F.cross_entropy(batch_mode_logits, best_mode, reduction='mean')
            total_cls_loss += cls_loss * num_objects
            
            # Calculate additional metrics
            min_ade = traj_loss.min(dim=-1)[0]  # [num_objects]
            min_fde = traj_loss.min(dim=-1)[0]  # [num_objects]
            miss_rate = (min_fde > 2.0).float()  # [num_objects]
            
            # Get probabilities for best mode
            best_mode_probs = F.softmax(batch_mode_logits, dim=-1)[torch.arange(num_objects), best_mode]
            brier_fde = min_fde + (1 - best_mode_probs) ** 2
            
            # Store metrics for overall calculation
            all_min_ade.append(min_ade)
            all_min_fde.append(min_fde)
            all_miss_rate.append(miss_rate)
            all_brier_fde.append(brier_fde)
            
            # Log metrics for this batch
            self.log(f'{status}/batch_{batch_idx}/minADE6', min_ade.mean(), on_step=True, on_epoch=False)
            self.log(f'{status}/batch_{batch_idx}/minFDE6', min_fde.mean(), on_step=True, on_epoch=False)
            self.log(f'{status}/batch_{batch_idx}/miss_rate', miss_rate.mean(), on_step=True, on_epoch=False)
            self.log(f'{status}/batch_{batch_idx}/brier_fde', brier_fde.mean(), on_step=True, on_epoch=False)
        
        # Calculate and log average losses
        avg_cls_loss = total_cls_loss / total_objects
        avg_reg_loss = total_reg_loss / total_objects
        total_loss = cls_coef * avg_cls_loss + reg_coef * avg_reg_loss
        
        self.log(f'{status}/cls_loss', avg_cls_loss, on_step=False, on_epoch=True)
        self.log(f'{status}/reg_loss', avg_reg_loss, on_step=False, on_epoch=True)
        self.log(f'{status}/total_loss', total_loss, on_step=False, on_epoch=True)
        
        # Calculate and log overall metrics
        all_min_ade = torch.cat(all_min_ade)
        all_min_fde = torch.cat(all_min_fde)
        all_miss_rate = torch.cat(all_miss_rate)
        all_brier_fde = torch.cat(all_brier_fde)
        
        self.log(f'{status}/minADE6', all_min_ade.mean(), on_step=False, on_epoch=True)
        self.log(f'{status}/minFDE6', all_min_fde.mean(), on_step=False, on_epoch=True)
        self.log(f'{status}/miss_rate', all_miss_rate.mean(), on_step=False, on_epoch=True)
        self.log(f'{status}/brier_fde', all_brier_fde.mean(), on_step=False, on_epoch=True)
        
        # Log additional metrics by dataset
        if status == 'val' and self.config.get('eval', False):
            # Get dataset names
            dataset_names = batch['input_dict']['dataset_name']
            unique_dataset_names = np.unique(dataset_names)
            
            # Log metrics by dataset
            for dataset_name in unique_dataset_names:
                dataset_mask = dataset_names == dataset_name
                if dataset_mask.any():
                    self.log(f'{status}/{dataset_name}/minADE6', all_min_ade[dataset_mask].mean(), on_step=False, on_epoch=True)
                    self.log(f'{status}/{dataset_name}/minFDE6', all_min_fde[dataset_mask].mean(), on_step=False, on_epoch=True)
                    self.log(f'{status}/{dataset_name}/miss_rate', all_miss_rate[dataset_mask].mean(), on_step=False, on_epoch=True)
                    self.log(f'{status}/{dataset_name}/brier_fde', all_brier_fde[dataset_mask].mean(), on_step=False, on_epoch=True)
            
            # Log metrics by trajectory type
            # trajectory_types = batch['input_dict']['trajectory_type']
            # trajectory_correspondance = {
            #     0: "stationary", 1: "straight", 2: "straight_right",
            #     3: "straight_left", 4: "right_u_turn", 5: "right_turn",
            #     6: "left_u_turn", 7: "left_turn"
            # }
            
            # for traj_type, traj_name in trajectory_correspondance.items():
            #     type_mask = trajectory_types == traj_type
            #     if type_mask.any():
            #         self.log(f'{status}/traj_type/{traj_name}/minADE6', all_min_ade[type_mask].mean(), on_step=False, on_epoch=True)
            #         self.log(f'{status}/traj_type/{traj_name}/minFDE6', all_min_fde[type_mask].mean(), on_step=False, on_epoch=True)
            #         self.log(f'{status}/traj_type/{traj_name}/miss_rate', all_miss_rate[type_mask].mean(), on_step=False, on_epoch=True)
            #         self.log(f'{status}/traj_type/{traj_name}/brier_fde', all_brier_fde[type_mask].mean(), on_step=False, on_epoch=True)
            
            # # Log metrics by agent type
            # agent_types = batch['input_dict']['center_objects_type']
            # agent_type_dict = {1: "vehicle", 2: "pedestrian", 3: "bicycle"}
            
            # for agent_type, agent_name in agent_type_dict.items():
            #     type_mask = agent_types == agent_type
            #     if type_mask.any():
            #         self.log(f'{status}/agent_type/{agent_name}/minADE6', all_min_ade[type_mask].mean(), on_step=False, on_epoch=True)
            #         self.log(f'{status}/agent_type/{agent_name}/minFDE6', all_min_fde[type_mask].mean(), on_step=False, on_epoch=True)
            #         self.log(f'{status}/agent_type/{agent_name}/miss_rate', all_miss_rate[type_mask].mean(), on_step=False, on_epoch=True)
            #         self.log(f'{status}/agent_type/{agent_name}/brier_fde', all_brier_fde[type_mask].mean(), on_step=False, on_epoch=True)
