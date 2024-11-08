import os
import sys
import torch
import argparse
from utils.custom_arg_parser import DefaultArgs


def compile_kernels():

    import os
    import sys
    import torch

    # Check if distributed mode is initialized and get the rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0 # Assume rank 0 if not distributed

    # Only execute on GPU with rank 0
    if rank == 0:
        os.system("echo '\n#################################################################'")
        os.system(f"echo 'Compilation of CUDA kernels on device with rank {rank}'")
        os.system("echo '#################################################################'")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Executing <pwd> command:'")
        os.system("pwd")
        os.system("echo '#################################################################'")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Executing <setup.py>:'")
        os.system("echo '#################################################################'")

        # Convert line endings in make.sh to Unix format
        # prefix "b" indicates binary mode
        with open("models/ops/make.sh", "rb") as f:
            content = f.read()
        with open("models/ops/make.sh", "wb") as f:
            f.write(content.replace(b"\r\n", b"\n"))

        os.system("cd ./models/ops && chmod +x make.sh && ./make.sh")

        os.system("echo '#################################################################'")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Available packages in active environment:'")
        os.system("echo '#################################################################'")
        os.system("conda list")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Running unit test script <test.py>:'")
        os.system("echo '#################################################################'")
        os.system("python models/ops/test.py")
        os.system("echo '#################################################################'")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Test 1 for import of package <MultiScaleDeformableAttention>:'")
        os.system("echo '#################################################################'")
        os.system("python models/ops/functions/test_import_1.py")
        os.system("echo '#################################################################'")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Test 2 for import of package <MultiScaleDeformableAttention>:'")
        os.system("echo '#################################################################'")
        os.system("python models/ops/test_import_2.py")
        os.system("echo '#################################################################'")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Test 3 for import of package <MultiScaleDeformableAttention>:'")
        os.system("echo '#################################################################'")
        os.system("python models/test_import_3.py")
        os.system("echo '#################################################################'")

        os.system("echo '\n#################################################################'")
        os.system("pip show MultiScaleDeformableAttention")
        os.system("echo '#################################################################'")

    else:
        os.system("echo '\n#################################################################'")
        os.system(f"echo 'Skipping compilation of CUDA kernels on device with rank {rank}'")
        os.system("echo '#################################################################'")

    # Synchronize all GPUs to wait until rank 0 completes the above commands
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_helper_parser():

    parser = argparse.ArgumentParser('Helper main script for screw detection', add_help=False)

    parser.add_argument('--lr', default=DefaultArgs.lr, type=float)
    parser.add_argument('--lr_backbone', default=DefaultArgs.lr_backbone, type=float)
    parser.add_argument('--batch_size', default=DefaultArgs.batch_size, type=int)
    parser.add_argument('--epochs', default=DefaultArgs.epochs, type=int)
    # parser.add_argument('--lr_drop_epochs', default=DefaultArgs.lr_drop_epochs, type=int, nargs='+') # Original code
    parser.add_argument('--lr_drop_epochs', default=DefaultArgs.lr_drop_epochs, type=int) # Adapted code

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=DefaultArgs.with_box_refine, action='store_true')
    parser.add_argument('--two_stage', default=DefaultArgs.two_stage, action='store_true')

    # Backbone
    parser.add_argument('--backbone', default=DefaultArgs.backbone, type=str,
                        help="Name of the convolutional backbone to use")

    # Transformer
    parser.add_argument('--num_queries', default=DefaultArgs.num_queries, type=int,
                        help="Number of query slots")
    
    # Efficient DETR
    parser.add_argument('--eff_query_init', default=DefaultArgs.eff_query_init, action='store_true')
    parser.add_argument('--eff_specific_head', default=DefaultArgs.eff_specific_head, action='store_true')

    # Sparse DETR
    parser.add_argument('--use_enc_aux_loss', default=DefaultArgs.use_enc_aux_loss, action='store_true')
    parser.add_argument('--rho', default=DefaultArgs.rho, type=float)

    # Alpha encoding
    parser.add_argument('--alpha_correspondence', default=DefaultArgs.alpha_correspondence, action='store_true')

    # Dataset parameters
    parser.add_argument('--data_dir', default=DefaultArgs.data_dir, type=str)
    parser.add_argument('--dataset_reduction', default=DefaultArgs.dataset_reduction, type=int)
    parser.add_argument('--num_workers', default=DefaultArgs.num_workers, type=int,
                        help='number of data loading workers')

    # Logging parameters
    parser.add_argument('--log_wandb', default=DefaultArgs.log_wandb, action='store_true')

    # Checkpointing
    parser.add_argument('--checkpoint_file', default=None, type=str,
                        help='path to checkpoint file')

    # only needed for medical backbones
    parser.add_argument('--backbone_checkpoint_file', default=None, type=str,
                        help='path to backbone checkpoint file')
    parser.add_argument('--result_dir', default=DefaultArgs.result_dir, type=str,
                        help='directory to store results like logs and checkpoints')

    # Additionally added
    parser.add_argument('--dataset_name', default="V1-1to3objects-400projections-circular", type=str, help='Dataset name')
    parser.add_argument('--job_ID', default="Test_Job", type=str, help='Unique job ID')
    parser.add_argument('--num_gpus', default=DefaultArgs.num_gpus, type=int, help='Number of GPUs per node')
    parser.add_argument('--num_nodes', default=DefaultArgs.num_nodes, type=int, help='Number of available nodes')                         

    return parser


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Helper main script for screw detection', parents=[get_helper_parser()])
    args = parser.parse_args()

    # Compile custom CUDA kernels
    compile_kernels()

    # Execute main_azure.py
    command = f"python models/main_azure.py --azure"
    for k, v in vars(args).items():
        if v is not None:
            if (type(v) == bool) and (v == False):
                command = command
            elif (type(v) == bool) and (v == True):
                command = command + f" --{k}"
            else:
                command = command + f" --{k} {v}"
    os.system(command)
