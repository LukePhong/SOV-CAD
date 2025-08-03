import torch
import torch.nn as nn

class CADEmbedding(nn.Module):
    """Enhanced embedding module for CAD sequences with support for both discrete and continuous parameters."""
    def __init__(self, cfg, use_group=False, group_len=None):
        super().__init__()
        
        self.use_enhanced_embedding = getattr(cfg, 'use_enhanced_embedding', False)
        self.use_continuous_params = getattr(cfg, 'use_continuous_params', False)
        self.model_dim = cfg.model_dim
        self.n_args = cfg.n_args
        
        # Command embedding (always discrete)
        self.command_embed = nn.Embedding(cfg.n_commands, cfg.model_dim)
        nn.init.xavier_uniform_(self.command_embed.weight)
        
        # Parameter embedding/projection
        if self.use_continuous_params:
            # Continuous parameters version
            # Direct projection for continuous values
            self.args_projection = nn.Sequential(
                nn.Linear(cfg.n_args, cfg.model_dim // 2),
                nn.LayerNorm(cfg.model_dim // 2),
                nn.ReLU(),
                nn.Linear(cfg.model_dim // 2, cfg.model_dim),
                nn.LayerNorm(cfg.model_dim)
            )
        else:
            # Discrete parameters version
            args_dim = cfg.args_dim + 1  # +1 for padding
            
            if self.use_enhanced_embedding:
                # Enhanced implementation
                self.arg_embed = nn.Embedding(args_dim, cfg.model_dim // 4, padding_idx=0)
                nn.init.xavier_uniform_(self.arg_embed.weight)
                
                # More sophisticated parameter projection
                self.param_projection = nn.Sequential(
                    nn.Linear(cfg.model_dim // 4 * cfg.n_args, cfg.model_dim),
                    nn.LayerNorm(cfg.model_dim),
                    nn.ReLU()
                )
            else:
                # Original implementation
                self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
                self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.model_dim)
        
        # Position encoding
        self.use_positional_encoding = getattr(cfg, 'use_positional_encoding', False)
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncodingLUT(cfg.model_dim, max_len=cfg.max_ep)
        
        if self.use_enhanced_embedding:
            # Output normalization
            self.output_norm = nn.LayerNorm(cfg.model_dim)
            
            # Dropout for regularization
            self.dropout = nn.Dropout(p=0.1)
        
        # Group embedding
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.model_dim)
            nn.init.xavier_uniform_(self.group_embed.weight)

    def forward(self, commands, args, groups=None):
        """
        Args:
            commands: Command tokens [S, N]
            args: Parameter values for each command [S, N, n_args]
                - For discrete mode: int tensor of token indices
                - For continuous mode: float tensor of continuous values
            groups: Optional group IDs [S, N]
        Returns:
            Embedded representation [S, N, model_dim]
        """
        S, N = commands.shape
        
        # Command embedding (always discrete)
        cmd_embedding = self.command_embed(commands.long())
        
        if self.use_continuous_params:
            # Continuous parameters handling
            # Process continuous parameter values directly
            args_projection = self.args_projection(args.float())
            
            # Combine embeddings
            src = cmd_embedding + args_projection
        else:
            # Discrete parameters handling
            if self.use_enhanced_embedding:
                # Enhanced implementation for discrete parameters
                args_embedding = self.arg_embed((args + 1).long())
                
                # Project the flattened parameter embeddings
                args_embedding_flat = args_embedding.view(S, N, -1)
                args_projection = self.param_projection(args_embedding_flat)
                
                # Combine embeddings
                src = cmd_embedding + args_projection
            else:
                # Original implementation for discrete parameters
                src = cmd_embedding + \
                      self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))
        
        # Apply group embedding if specified
        if self.use_group and groups is not None:
            group_embedding = self.group_embed(groups.long())
            src = src + group_embedding
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            src = self.pos_encoding(src)
            
        # Apply output normalization and dropout
        if self.use_enhanced_embedding:
            src = self.output_norm(src)
            src = self.dropout(src)
        
        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.model_dim = cfg.model_dim
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.model_dim, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.model_dim))
        return src

class PositionalEncodingLUT(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=250, repeat = 1):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)

        self.pos_embed = nn.Embedding(max_len, model_dim)

        self._init_embeddings()

        self.repeat = repeat

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x, repeat=1, start=0):
        pos = self.position[start:x.size(1)+start].flatten()
        if repeat == 1:
            pos_ebd = self.pos_embed(pos)
        else:
            pos_ebd = self.pos_embed(pos).unsqueeze(1)
            pos_ebd = pos_ebd.expand(-1, repeat, -1)
        x = x + pos_ebd
        return self.dropout(x)

