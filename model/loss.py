import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import _get_padding_mask, _get_visibility_mask
from cadlib.macro import *


class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights
        self.use_continuous_params = getattr(cfg, 'use_continuous_params', False)
        self.param_loss_type = getattr(cfg, 'param_loss_type', 'mse')

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, outputs, tgt_commands, tgt_args):
        """
        Args:
            outputs: Dict containing model outputs
                - For discrete params: dict with 'command_logits' and 'args_logits'
                - For continuous params: dict with 'command_logits' and 'args_values'
            tgt_commands: Target command tokens
            tgt_args: Target parameter values
                - For discrete mode: int tensor of token indices
                - For continuous mode: float tensor of continuous values
        """
        # Teacher forcing
        # rotate the tgt_commands and tgt_args, change the last element to EOS
        tgt_commands = torch.cat((tgt_commands[:, 1:], EOS_IDX * torch.ones_like(tgt_commands[:, :1])), dim=1)
        tgt_args = torch.cat((tgt_args[:, 1:], torch.tensor(EOS_VEC[1:]).to(tgt_args.device) * torch.ones_like(tgt_args[:, :1, :])), dim=1)

        # Get masks
        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)    
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)   
        arg_mask = self.cmd_args_mask[tgt_commands.long()]

        # Command loss is always cross-entropy (commands are always discrete)
        command_logits = outputs['command_logits']
        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()], tgt_commands[padding_mask.bool()].long())

        # Parameter loss depends on continuous vs discrete mode
        if self.use_continuous_params:
            # Continuous parameter loss using regression
            args_values = outputs['args_values']
            valid_args = arg_mask.bool()
            
            if self.param_loss_type == 'mse':
                loss_args = F.mse_loss(
                    args_values[valid_args], 
                    tgt_args[valid_args].float()
                )
            elif self.param_loss_type == 'l1':
                loss_args = F.l1_loss(
                    args_values[valid_args], 
                    tgt_args[valid_args].float()
                )
            elif self.param_loss_type == 'smooth_l1':
                loss_args = F.smooth_l1_loss(
                    args_values[valid_args], 
                    tgt_args[valid_args].float()
                )
            else:
                raise ValueError(f"Unknown param_loss_type: {self.param_loss_type}")
        else:
            # Discrete parameter loss using cross-entropy
            args_logits = outputs['args_logits']
            loss_args = F.cross_entropy(
                args_logits[arg_mask.bool()], 
                tgt_args[arg_mask.bool()].long() + 1  # shift due to -1 PAD_VAL
            )

        # Weight the losses
        total_weight = self.weights["loss_cmd_weight"] + self.weights["loss_args_weight"]
        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd / total_weight
        loss_args = self.weights["loss_args_weight"] * loss_args / total_weight

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res
