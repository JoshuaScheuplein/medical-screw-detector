
print("\nEntered 'main_azure.py' ...")

print("\nStarting to import necessary packages ...")

import os
import argparse
import json # Additionally added
import signal
from pathlib import Path # Additionally added

import wandb
from azureml.core import Run # Additionally added

import time # Additionally added
from datetime import timedelta # Additionally added

import torch
import torchvision # Additionally added to solve import issue ...
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything # Additionally added
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback # Additionall added
from pytorch_lightning.utilities.rank_zero import rank_zero_only # Additionally added

#######################################################################################
try:
    import MultiScaleDeformableAttention as MSDA
    print("\n'MultiScaleDeformableAttention' is located at:", MSDA.__file__)
except Exception as e:
    print("\nFailed to import 'MultiScaleDeformableAttention' in main_azure.py")
    print(e)
#######################################################################################

#######################################################################################
# Additionally added PYTHONPATH since main_azure.py is located in folder 'models'
import sys
print("\nPYTHONPATH 1 in main_azure.py:", sys.path)
path = os.path.dirname(os.path.dirname(__file__))
print("Adding path to PYTHONPATH:", path)
sys.path.append(path)
print("PYTHONPATH 2 in main_azure.py:", sys.path)
#######################################################################################

from dataset.DatasetBuilder import build_dataset, custom_collate_fn
from lightning_copy.prediction_logging_callback import PredictionLoggingCallback
from lightning_copy.detr_model import DeformableDETRLightning
from utils.custom_arg_parser import get_args_parser

print("\nSuccessfully imported all required packages!")

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
    @rank_zero_only # Additionally added
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = time.time()
        progress_logger.info(f"\nStarting training epoch {trainer.current_epoch + 1} ...")
        # print(f"\nStarting training epoch {trainer.current_epoch + 1} ...")

    @rank_zero_only # Additionally added
    def on_train_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.train_epoch_start_time
        readable_time = self.format_time(elapsed_time)
        progress_logger.info(f"Training epoch {trainer.current_epoch + 1} completed in {readable_time}")
        # print(f"Training epoch {trainer.current_epoch + 1} completed in {readable_time}")

    # Training Batch Timing
    # @rank_zero_only # Additionally added
    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     self.batch_start_time = time.time()

    # @rank_zero_only # Additionally added
    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     batch_elapsed = time.time() - self.batch_start_time
    #     print(f"Training batch {batch_idx + 1} took {self.format_time(batch_elapsed)}")

    # Validation Epoch Timing
    @rank_zero_only # Additionally added
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start_time = time.time()
        progress_logger.info(f"Starting validation epoch {trainer.current_epoch + 1} ...")
        # print(f"Starting validation epoch {trainer.current_epoch + 1}...")

    @rank_zero_only # Additionally added
    def on_validation_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.val_epoch_start_time
        readable_time = self.format_time(elapsed_time)
        progress_logger.info(f"Validation epoch {trainer.current_epoch + 1} completed in {readable_time}")
        # print(f"Validation epoch {trainer.current_epoch + 1} completed in {readable_time}")

    # Validation Batch Timing
    # @rank_zero_only # Additionally added
    # def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     self.batch_start_time = time.time()

    # @rank_zero_only # Additionally added
    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     batch_elapsed = time.time() - self.batch_start_time
    #     print(f"Validation batch {batch_idx + 1} took {self.format_time(batch_elapsed)}")

    # Test Epoch Timing
    @rank_zero_only # Additionally added
    def on_test_epoch_start(self, trainer, pl_module):
        self.test_epoch_start_time = time.time()
        progress_logger.info("\nStarting test epoch ...")
        # print("\nStarting test epoch...")

    @rank_zero_only # Additionally added
    def on_test_epoch_end(self, trainer, pl_module):
        elapsed_time = time.time() - self.test_epoch_start_time
        readable_time = self.format_time(elapsed_time)
        progress_logger.info(f"Test epoch completed in {readable_time}")
        # print(f"Test epoch completed in {readable_time}")

    # Test Batch Timing
    # @rank_zero_only # Additionally added
    # def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     self.batch_start_time = time.time()

    # @rank_zero_only # Additionally added
    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     batch_elapsed = time.time() - self.batch_start_time
    #     print(f"Test batch {batch_idx + 1} took {self.format_time(batch_elapsed)}")


def main(args, print_flag):

    #########################
    # set seeds for numpy, torch, python.random and PYTHONHASHSEED
    # (Additionally added)
    #########################

    seed_everything(42, workers=True)

    #########################
    # save settings
    # (Additionally added)
    #########################

    args.lr = args.lr * args.num_gpus * args.num_nodes
    args.lr_backbone = args.lr_backbone * args.num_gpus * args.num_nodes

    if print_flag:
        output_file = Path(args.result_dir) / f"{args.job_ID}_settings.json"
        with open(output_file, 'w') as f:
            json.dump(vars(args), f, indent=4)

    #########################
    # init logger
    #########################

    if args.log_wandb:
        try:
            api_key = os.environ["WANDB_API_KEY"]
            wandb.login(key=api_key)
            print(f"\nSuccessfully logged in to W&B with API key '{api_key}'")
        except Exception as e:
            print(f"\nCould not login to W&B service: {e}")

        if args.use_enc_aux_loss:
            wandb_run_identifier = f"Sparse_DETR_{args.job_ID}"
        elif args.eff_query_init:
            wandb_run_identifier = f"Efficient_DETR_{args.job_ID}"
        else:
            wandb_run_identifier = f"Deformable_DETR_{args.job_ID}"

        logger = WandbLogger(project="MedDINO",
                             group="Screw-Detection", # organize individual runs into a larger experiment
                             entity="joshua-scheuplein", # username
                             name=wandb_run_identifier, # general job descriptor
                             id=wandb_run_identifier, # unique job descriptor
                             config=args, # save settings and hyperparameters
                             save_dir=args.result_dir, # where to store wandb files
                             save_code=True, # save main script
                             resume="allow", # needed in case of preempted job
                             )  
    else:
        # logger = CSVLogger("logs", name="local_log") # Original
        logger = CSVLogger(save_dir=args.result_dir, flush_logs_every_n_steps=1) # Adapted

    run = Run.get_context() # Setup for Azure logging and job resuming

    #########################
    # init dataloader
    #########################

    if print_flag:
        print("\nBatchsize:", args.batch_size)
        print("Num Workers:", args.num_workers)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)
    
    #####################################################################################################
    # Original code
    #####################################################################################################
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=custom_collate_fn, num_workers=args.num_workers,
    #                                pin_memory=False, persistent_workers=True)

    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
    #                              collate_fn=custom_collate_fn, num_workers=args.num_workers,
    #                              pin_memory=False, persistent_workers=True)

    # data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False,
    #                               collate_fn=custom_collate_fn, num_workers=args.num_workers,
    #                               persistent_workers=True)

    #####################################################################################################
    # Original code
    #####################################################################################################
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, drop_last=True,
                                   collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                   pin_memory=False, persistent_workers=True)

    data_loader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, drop_last=False,
                                 collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                 pin_memory=False, persistent_workers=True)

    data_loader_test = DataLoader(dataset_test, args.batch_size, shuffle=False, drop_last=False,
                                  collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  persistent_workers=True)
    #####################################################################################################

    if print_flag:
        print(f"\nNumber of batches in 'Train' Dataloader: {len(data_loader_train)}")
        print(f"Number of batches in 'Val' Dataloader: {len(data_loader_val)}")
        print(f"Number of batches in 'Test' Dataloader: {len(data_loader_test)}")

    #########################
    # init callbacks
    #########################

    checkpoint_dir = os.path.join(args.result_dir, "Checkpoints")
    # os.makedirs(checkpoint_dir, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=False, exist_ok=True)

    # Additionally added
    checkpoint_last_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath=checkpoint_dir,
        filename="backup_checkpoint_{epoch:02d}",
        save_last=False, # saves a last.ckpt copy whenever a checkpoint file gets saved
    )
    
    # checkpoint_last_callback = ModelCheckpoint(
    #     save_top_k=1,
    #     monitor="epoch",
    #     mode="max",
    #     dirpath=checkpoint_dir,
    #     filename="backup_checkpoint",
    #     save_last=False, # saves a last.ckpt copy whenever a checkpoint file gets saved
    # )

    # saves top-K checkpoints based on "train_loss" metric
    checkpoint_train_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename="checkpoint-training-{epoch:02d}-{train_loss:.2f}",
        save_last=False
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_val_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename="checkpoint-validation-{epoch:02d}-{val_loss:.2f}",
        save_last=False,
    )

    for dataset in [dataset_train, dataset_val, dataset_test]:
        for volume_name in dataset.volume_names:
            prediction_path = os.path.join(args.result_dir, args.dataset_name, volume_name)
            # os.makedirs(prediction_path, exist_ok=True)
            Path(prediction_path).mkdir(parents=True, exist_ok=True)

    prediction_logging_callback = PredictionLoggingCallback(args.result_dir, batch_size=args.batch_size)
    epoch_logging_callback = EpochLoggingCallback() # Additionally added

    #########################
    # Train the Model
    #########################

    detr_model = DeformableDETRLightning(args)

    # if os.name == 'nt':
    #     plugins = []
    # else:
    #     print("Using SLURM environment plugin!\n")
    #     plugins = [SLURMEnvironment(requeue_signal=signal.SIGUSR1)]

    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      #####################################################################################
                      # Adapted for multi-GPU training
                      #####################################################################################
                      accelerator="gpu", # Enable GPU acceleration
                      # strategy= "ddp",
                      # RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
                      strategy="ddp_find_unused_parameters_true", # Distributed Data Parallel strategy
                      use_distributed_sampler=True,
                      devices=args.num_gpus,
                      num_nodes=args.num_nodes,
                      sync_batchnorm=True,
                      #####################################################################################
                      default_root_dir=args.result_dir,
                      log_every_n_steps=100,
                      # callbacks=[checkpoint_val_callback, prediction_logging_callback], # Original Code
                      callbacks=[checkpoint_last_callback, checkpoint_val_callback, checkpoint_train_callback,
                                 prediction_logging_callback, epoch_logging_callback], # Adapted Code
                      enable_progress_bar=False, # Additionally added
                      # plugins=plugins, # We do not need any plugins on Azure
                      )

    backup_checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("backup_checkpoint_epoch="):
            backup_checkpoints.append(filename)
    if len(backup_checkpoints) > 0:
        last_ckpt_file = os.path.join(checkpoint_dir, sorted(backup_checkpoints)[-1])
    else:
        last_ckpt_file = "None"

    if (args.checkpoint_file is None) and (os.path.isfile(last_ckpt_file)):
        args.checkpoint_file = last_ckpt_file
        if print_flag:
            print(f"Resume training from checkpoint '{args.checkpoint_file}'\n")
    elif args.checkpoint_file is not None:
        if print_flag:
            print(f"Resume training from manually specified checkpoint '{args.checkpoint_file}'\n")
    else:
        if print_flag:
            print(f"Starting a complete new training run WITHOUT any pretrained model checkpoint ...\n")
    
    # last_ckpt_file = os.path.join(checkpoint_dir, "backup_checkpoint.ckpt")
    # if (args.checkpoint_file is None) and (os.path.isfile(last_ckpt_file)):
    #     if print_flag:
    #         print(f"Resume training from checkpoint '{last_ckpt_file}'\n")
    #     args.checkpoint_file = last_ckpt_file
    # else:
    #     if print_flag:
    #         print(f"Starting a complete new training run WITHOUT any pretrained model checkpoint ...\n")

    trainer.fit(model=detr_model,
                train_dataloaders=data_loader_train,
                val_dataloaders=data_loader_val,
                ckpt_path=args.checkpoint_file)

    #########################
    # Test the Model
    #########################

    trainer.test(model=detr_model,
                 dataloaders=data_loader_test)


if __name__ == '__main__':

    # Check if distributed mode is initialized and get the rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0 # Assume rank 0 if not distributed

    if rank == 0:
        print_flag = True
        print(f"\nSetting print_flag = '{print_flag}' on device with rank {rank}")
    else:
        print_flag = False
        print(f"\nSetting print_flag = '{print_flag}' on device with rank {rank}")
    
    parser = argparse.ArgumentParser('Deformable DETR Detector for implants', parents=[get_args_parser()])
    args = parser.parse_args()

    if print_flag:
        print("\nEntered main method of main_azure.py script ...")
        for k, v in vars(args).items():
            print(f"{k} = {v}")

    #########################
    # create result_dir
    #########################
    try:
        os.mkdir(args.result_dir)
        # Path(args.result_dir).mkdir(parents=False, exist_ok=False)
        print(f"\nResult directory '{args.result_dir}' created successfully.")
    except FileExistsError:
        print(f"\nResult directory '{args.result_dir}' already exists.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    #########################
    # start model training
    #########################
    main(args, print_flag)
