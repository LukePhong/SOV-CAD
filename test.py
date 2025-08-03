import faulthandler
faulthandler.enable()
import random
import sys
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pytorch_lightning import loggers as pl_loggers
from model.model import TransformerModel
from model.data import get_dataloaders
import argparse
import json
import torch
from model.config import Config  
import time

from torch.utils.data import Dataset, DataLoader
from model.data import ViewBuilderDataset

def main(cfg):

    # Load test files based on test_mode
    to_test = []
    
    if cfg.test_mode == "json":
        # Load data file paths from JSON dataset file
        with open(cfg.dataset_file, 'r') as f:
            dataset_file = json.load(f)   
        to_test = dataset_file['test']
    
    elif cfg.test_mode == "folder":
        # Test all step files in the specified folder
        test_file_path = Path(cfg.step_path)
        to_test = [f.stem for f in test_file_path.iterdir() if f.suffix == ".step"]

        # # random select 1000 files
        # to_test = random.sample(to_test, 1000)
        # sort to_test
        to_test.sort()
        print(f"Testing {len(to_test)} files from folder: {cfg.step_path}")
    
    elif cfg.test_mode == "file_list":
        # Load list of files from a text file
        with open(cfg.test_file_list, "r") as f:
            to_test = [line.strip() for line in f.readlines()]
        print(f"Testing {len(to_test)} files from list: {cfg.test_file_list}")
    
    else:
        raise ValueError(f"Invalid test_mode: {cfg.test_mode}. Must be one of: 'json', 'folder', 'file_list'")

    test_dataset = ViewBuilderDataset(to_test, cfg, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)


    print(f"Test Data Count: {len(test_dataloader.dataset)}")

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

    # Define Timer callback with a timeout for each epoch
    # timer_callback = Timer(duration="00:00:10:00")  

    # Test model
    trainer = pl.Trainer(
        # gpus=1 if torch.cuda.is_available() else 0,
        devices=cfg.gpu_idx,
        accelerator='gpu',
        # callbacks=[timer_callback]
    )
    rsl = trainer.test(model, test_dataloaders=test_dataloader, ckpt_path=cfg.checkpoint_path)
    print(rsl)
    print("Test finished")


if __name__ == '__main__':
    
    config = Config()

    main(config)