import argparse
import os
import signal
import torch
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

from dataset.DatasetBuilder import build_dataset, custom_collate_fn
from lightning.prediction_logging_callback import PredictionLoggingCallback
from lightning.detr_model import DeformableDETRLightning
from utils.custom_arg_parser import get_args_parser

########################################
import logging
logging.basicConfig(level=logging.DEBUG)
########################################


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
            wandb_run_identifier = f"Sparse_DETR_{args.backbone}_{job_name}_{args.job_ID}"
        elif args.eff_query_init:
            wandb_run_identifier = f"Efficient_DETR_{args.backbone}_{job_name}_{args.job_ID}"
        else:
            wandb_run_identifier = f"Deformable_DETR_{args.backbone}_{job_name}_{args.job_ID}"

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

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)

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

    print("\nBatchsize:", args.batch_size)
    print("Num Workers:", args.num_workers)

    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train, drop_last=False,
    #                                collate_fn=custom_collate_fn, num_workers=args.num_workers,
    #                                pin_memory=False, persistent_workers=False)

    # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False,
    #                              collate_fn=custom_collate_fn, num_workers=args.num_workers,
    #                              pin_memory=False, persistent_workers=False)

    # data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False,
    #                               collate_fn=custom_collate_fn, num_workers=args.num_workers,
    #                               pin_memory=False, persistent_workers=False)

    data_loader_train = DataLoader(dataset_train, args.batch_size, shuffle=True, drop_last=True,
                                   collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                   pin_memory=False, persistent_workers=True)

    data_loader_val = DataLoader(dataset_val, args.batch_size, shuffle=False, drop_last=False,
                                 collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                 pin_memory=False, persistent_workers=True)

    data_loader_test = DataLoader(dataset_test, args.batch_size, shuffle=False, drop_last=False,
                                  collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=False, persistent_workers=True)

    #####################################################################################################

    print(f"\nNumber of Samples in 'Train' Dataloader: {len(data_loader_train)}")
    print(f"\nNumber of Samples in 'Val' Dataloader: {len(data_loader_val)}")
    print(f"\nNumber of Samples in 'Test' Dataloader: {len(data_loader_test)}")

    #########################
    # init callbacks
    #########################

    # saves top-K checkpoints based on "train_loss" metric
    checkpoint_train_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        dirpath=args.result_dir,
        filename="sample-train_loss-{epoch:02d}-{train_loss:.2f}",
        save_last=True # saves a last.ckpt copy whenever a checkpoint file gets saved
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_val_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=args.result_dir,
        filename="sample-val_loss-{epoch:02d}-{val_loss:.2f}",
        save_last=False,
    )

    for dataset in [dataset_train, dataset_val, dataset_test]:
        for volume_name in dataset.volume_names:
            prediction_path = os.path.join(args.result_dir,
                                           os.path.basename(os.path.normpath(dataset.data_dir)),
                                           volume_name)
            os.makedirs(prediction_path, exist_ok=True)

    prediction_logging_callback = PredictionLoggingCallback(args.result_dir, batch_size=args.batch_size)

    #########################
    # Train the Model
    #########################

    detr_model = DeformableDETRLightning(args)

    if os.name == 'nt':
        plugins = []
    else:
        print("\nUsing SLURM environment ...")
        plugins = [SLURMEnvironment(requeue_signal=signal.SIGUSR1)]

    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      devices=1,
                      num_nodes=1,
                      default_root_dir=args.result_dir,
                      # log_every_n_steps=100, # How often to log within steps
                      log_every_n_steps=1, # How often to log within steps
                      # callbacks=[checkpoint_val_callback, prediction_logging_callback], # Original Code
                      # callbacks=[checkpoint_train_callback, prediction_logging_callback], # Adapted Code
                      # plugins=plugins,
                      enable_model_summary=True, # Enable detailed model summary (Additionally added)
                      )

    last_ckpt_file = args.result_dir + "/last.ckpt"
    if (args.checkpoint_file is None) and (os.path.isfile(last_ckpt_file)):
        args.checkpoint_file = last_ckpt_file

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
