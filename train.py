import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from model.model import TransformerModel
from model.data import get_dataloaders
import argparse
import json
import torch
from model.config import Config  
import time
import os
from pytorch_lightning.plugins import DDPPlugin

RESUME = False  # Set to True to resume training

def save_config_to_json(cfg, filepath):
    """Save config to a JSON file"""
    config_dict = {k: v for k, v in cfg.__dict__.items()}
    # Convert non-serializable objects (like GPU indices) to strings
    for key, value in config_dict.items():
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.device):
            config_dict[key] = [str(v) for v in value]
        elif isinstance(value, torch.device):
            config_dict[key] = str(value)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)

def main(cfg):
    # Load data file paths
    with open(cfg.dataset_file, 'r') as f:
        dataset_file = json.load(f)

    train_dataloader, val_dataloader = get_dataloaders(
        dataset_file['train'], 
        dataset_file['validation'], 
        cfg
    )

    # Define model
    model = TransformerModel(
        input_dim=cfg.input_dim,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        vocab_size=cfg.vocab_size,
        lr=cfg.lr,
        cfg=cfg
    )

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",
        dirpath=cfg.checkpoint_dir,
        save_top_k=1,
        mode="min",
    )

    # Define a path to save the logs which is based on date and time 
    month_day = time.strftime('%m%d')
    hour_min_second = time.strftime('%H%M%S')
    tb_logger = pl_loggers.TensorBoardLogger(
        cfg.log_dir,
        name=month_day,
        version=hour_min_second
    )
    
    # Save configuration to the checkpoint directory
    config_checkpoint_path = os.path.join(cfg.checkpoint_dir, 'config.json')
    save_config_to_json(cfg, config_checkpoint_path)

    # Train model
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        # gpus=1 if torch.cuda.is_available() else 0,
        devices=cfg.gpu_idx,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        logger=tb_logger,
        amp_backend='apex' if cfg.use_amp else 'native',
        strategy=DDPPlugin(find_unused_parameters=False) if len(cfg.gpu_idx) > 1 else None,
        # plugins='ddp_sharded',
        # fast_dev_run=True   
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.checkpoint_path if RESUME else None)


if __name__ == '__main__':
    
    config = Config()

    main(config)