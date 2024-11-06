import argparse
import os
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
    # init logger
    #########################

    if args.log_wandb:

        # wandb.login(key="fc47046192188490a1fcedc7a411218e15247c56")
        wandb.login(key="b8b1693523deba0245ee3284c25847b029261f90")

        job_name = args.backbone_checkpoint_file.split("/")[-1]
        assert "DINO_Training_" in job_name
        job_name = job_name.split("DINO_Training_")[-1]
        job_name = job_name.replace(".pth", "")

        if args.use_enc_aux_loss:
            wandb_run_identifier = f"Sparse_DETR_{args.job_ID}_{args.backbone}_{job_name}"
        elif args.eff_query_init:
            wandb_run_identifier = f"Efficient_DETR_{args.job_ID}_{args.backbone}_{job_name}"
        else:
            wandb_run_identifier = f"Deformable_DETR_{args.job_ID}_{args.backbone}_{job_name}"

        # logger = WandbLogger(project="Deformable DETR for dense image recognition",
        #                      name=wandb_run_identifier,
        #                      config=vars(args),
        #                      save_dir=args.result_dir)

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
        filename="backup_checkpoint",
        save_last=False, # saves a last.ckpt copy whenever a checkpoint file gets saved
    )

    # saves top-K checkpoints based on "train_loss" metric
    checkpoint_train_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        dirpath=checkpoint_dir, # Original: args.result_dir
        filename="checkpoint-training-{epoch:02d}-{train_loss:.2f}",
        save_last=False # Originally set to 'True'
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_val_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dir, # Original: args.result_dir
        filename="checkpoint-validation-{epoch:02d}-{val_loss:.2f}",
        save_last=False,
    )

    for dataset in [dataset_train, dataset_val, dataset_test]:
        for volume_name in dataset.volume_names:
            prediction_path = os.path.join(args.result_dir,
                                           os.path.basename(os.path.normpath(dataset.data_dir)),
                                           volume_name)
            os.makedirs(prediction_path, exist_ok=True)

    prediction_logging_callback = PredictionLoggingCallback(args.result_dir, batch_size=args.batch_size)
    epoch_logging_callback = EpochLoggingCallback() # Additionally added

    #########################
    # Train the Model
    #########################

    detr_model = DeformableDETRLightning(args)

    if os.name == 'nt':
        plugins = []
    else:
        print("Using SLURM environment plugin!")
        plugins = [SLURMEnvironment(requeue_signal=signal.SIGUSR1)]

    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      devices=1,
                      num_nodes=1,
                      default_root_dir=args.result_dir,
                      log_every_n_steps=100,
                      # callbacks=[checkpoint_val_callback, prediction_logging_callback], # Original Code
                      callbacks=[checkpoint_last_callback, checkpoint_val_callback, checkpoint_train_callback,
                                 prediction_logging_callback, epoch_logging_callback], # Adapted Code
                      enable_progress_bar=False, # Additionally added
                      plugins=plugins,
                      )

    # Original code
    # last_ckpt_file = args.result_dir + "/last.ckpt"
    # if (args.checkpoint_file is None) and (os.path.isfile(last_ckpt_file)):
    #     print(f"Resume training from checkpoint '{last_ckpt_file}'\n")
    #     args.checkpoint_file = last_ckpt_file

    # Adapted code
    last_ckpt_file = os.path.join(checkpoint_dir, "backup_checkpoint.ckpt")
    if (args.checkpoint_file is None) and (os.path.isfile(last_ckpt_file)):
        args.checkpoint_file = last_ckpt_file
        print(f"\nResume training from found checkpoint '{args.checkpoint_file}'\n")
    elif args.checkpoint_file is not None:
        print(f"\nResume training from manually specified checkpoint '{args.checkpoint_file}'\n")
    else:
        print(f"\nStarting a complete new training run WITHOUT any pretrained model checkpoint ...\n")

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
    
    parser = argparse.ArgumentParser('Deformable DETR Detector for implants', parents=[get_args_parser()])
    args = parser.parse_args()

    try:
        os.mkdir(args.result_dir)
        print(f"\nDirectory '{args.result_dir}' created successfully.")
    except FileExistsError:
        print(f"\nDirectory '{args.result_dir}' already exists.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    main(args)
