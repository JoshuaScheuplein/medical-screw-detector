import argparse
import os
import json # Additionally added
import signal
import torch
import wandb
import time # Additionally added
from datetime import timedelta # Additionally added
from pathlib import Path # Additionally added

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback # Additionall added

from torch.utils.data import DataLoader
from dataset.DatasetBuilder import build_dataset, custom_collate_fn

# from lightning.prediction_logging_callback import PredictionLoggingCallback       # Original code
# from lightning.detr_model import DeformableDETRLightning                          # Original code
from lightning_copy.prediction_logging_callback import PredictionLoggingCallback    # Adapted code
from lightning_copy.detr_model import DeformableDETRLightning                       # Adapted code

from utils.custom_arg_parser import get_args_parser

###############################################################
import logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO, format='%(message)s')
progress_logger = logging.getLogger(__name__)
###############################################################

# Additionally added
class EpochLoggingCallback(Callback):

    def __init__(self):
        super().__init__()
        self.train_epoch_start_time = None
        self.val_epoch_start_time = None
        self.test_epoch_start_time = None
        self.batch_start_time = None

    def format_time(self, elapsed_seconds):
        hours, remainder = divmod(int(elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((elapsed_seconds - int(elapsed_seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    # Training Epoch Timing
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = time.time()
        progress_logger.info(f"\nStarting training epoch {trainer.current_epoch + 1} ...")
        # print(f"\nStarting training epoch {trainer.current_epoch + 1} ...")

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.train_epoch_start_time
        readable_time = self.format_time(elapsed_time)
        progress_logger.info(f"Training epoch {trainer.current_epoch + 1} completed in {readable_time}")
        # print(f"Training epoch {trainer.current_epoch + 1} completed in {readable_time}")

    # Training Batch Timing
    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     self.batch_start_time = time.time()

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     batch_elapsed = time.time() - self.batch_start_time
    #     print(f"Training batch {batch_idx + 1} took {self.format_time(batch_elapsed)}")

    # Validation Epoch Timing
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start_time = time.time()
        progress_logger.info(f"Starting validation epoch {trainer.current_epoch + 1} ...")
        # print(f"Starting validation epoch {trainer.current_epoch + 1}...")

    def on_validation_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.val_epoch_start_time
        readable_time = self.format_time(elapsed_time)
        progress_logger.info(f"Validation epoch {trainer.current_epoch + 1} completed in {readable_time}")
        # print(f"Validation epoch {trainer.current_epoch + 1} completed in {readable_time}")

    # Validation Batch Timing
    # def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     self.batch_start_time = time.time()

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     batch_elapsed = time.time() - self.batch_start_time
    #     print(f"Validation batch {batch_idx + 1} took {self.format_time(batch_elapsed)}")

    # Test Epoch Timing
    def on_test_epoch_start(self, trainer, pl_module):
        self.test_epoch_start_time = time.time()
        progress_logger.info("\nStarting test epoch ...")
        # print("\nStarting test epoch...")

    def on_test_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.test_epoch_start_time
        readable_time = self.format_time(elapsed_time)
        progress_logger.info(f"Test epoch completed in {readable_time}")
        # print(f"Test epoch completed in {readable_time}")

    # Test Batch Timing
    # def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     self.batch_start_time = time.time()

    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     batch_elapsed = time.time() - self.batch_start_time
    #     print(f"Test batch {batch_idx + 1} took {self.format_time(batch_elapsed)}")


def main(args):

    #########################
    # init dataloader
    #########################

    print("\nBatchsize:", args.batch_size)
    print("Num Workers:", args.num_workers)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                   pin_memory=False, persistent_workers=True)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
                                 collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                 pin_memory=False, persistent_workers=True)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False,
                                  collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  persistent_workers=True)

    print(f"\nNumber of batches in 'Train' Dataloader: {len(data_loader_train)}")
    print(f"Number of batches in 'Val' Dataloader: {len(data_loader_val)}")
    print(f"Number of batches in 'Test' Dataloader: {len(data_loader_test)}")

    #########################
    # Load the model
    #########################

    detr_model = DeformableDETRLightning(args)

    if os.name == 'nt':
        plugins = []
    else:
        print("\nUsing SLURM environment plugin!")
        plugins = [SLURMEnvironment(requeue_signal=signal.SIGUSR1)]

    trainer = Trainer(devices=1,
                      num_nodes=1,
                      enable_progress_bar=True, # Additionally added
                      plugins=plugins,
                      detect_anomaly=True # Additionally added
                      )

    assert os.path.isfile(args.checkpoint_file), "Could not find model checkpoint!"
    print(f"\nTestset evaluation for checkpoint '{args.checkpoint_file}'\n")

    #########################
    # Test the Model
    #########################

    trainer.test(model=detr_model,
                 dataloaders=data_loader_test,
                 ckpt_path=args.checkpoint_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Testset evaluation for screw detection task', add_help=False)
    parser.add_argument('--settings_file', required=True, type=str)
    parser.add_argument('--checkpoint_file', required=True, type=str)
    test_args = parser.parse_args()
    test_args_dict = vars(test_args)

    assert os.path.isfile(test_args.settings_file), "Could not find .json file containing model settings!"
    with open(test_args.settings_file, "r") as f:
        train_args_dict = json.load(f)
        train_args_dict.pop('checkpoint_file', None) # Ensure that only checkpoint_file for test set is used ...
    
    merged_args_dict = {**train_args_dict, **test_args_dict}
    args = argparse.Namespace(**merged_args_dict)

    main(args)
