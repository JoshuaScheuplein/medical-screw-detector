import os
import argparse
from utils.custom_arg_parser import get_args_parser


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
        
        print("PYTHONPATH Old:", sys.path)

        os.system("echo '\nExecuting <pwd> command:'")
        os.system("pwd")

        os.system("echo '\n#################################################################'")

        os.system("echo '\nExecuting <ls> command:'")
        os.system("ls")

        os.system("echo '\nExecuting <ls models/ops> command:'")
        os.system("ls models/ops")

        os.system("echo '\nExecuting <ls models/ops/functions> command:'")
        os.system("ls models/ops/functions")

        os.system("echo '\nExecuting <ls models/ops/modules> command:'")
        os.system("ls models/ops/modules")

        os.system("echo '\n#################################################################'")
        os.system("echo 'Executing <setup.py>:'")
        os.system("echo '#################################################################'")

        # os.system("python models/ops/setup.py build install")

        # Convert line endings in make.sh to Unix format
        # prefix "b" indicates binary mode
        with open("models/ops/make.sh", "rb") as f:
            content = f.read()
        with open("models/ops/make.sh", "wb") as f:
            f.write(content.replace(b"\r\n", b"\n"))

        os.system("cd ./models/ops && chmod +x make.sh && ./make.sh")

        os.system("echo '#################################################################'")

        os.system("echo '\nExecuting <ls> command:'")
        os.system("ls")

        os.system("echo '\nExecuting <ls models/ops> command:'")
        os.system("ls models/ops")

        os.system("echo '\nExecuting <ls models/ops/functions> command:'")
        os.system("ls models/ops/functions")

        os.system("echo '\nExecuting <ls models/ops/modules> command:'")
        os.system("ls models/ops/modules")

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

        ######################################################################################
        import os
        import sys
        # path = os.path.join(os.path.dirname(__file__), "models")
        # path = os.path.dirname(os.path.dirname(__file__))
        path = os.path.dirname(__file__)
        print("Adding path to PYTHONPATH:", path)
        sys.path.append(path) # Adds the current directory tp $PYTHONPATH
        ######################################################################################

        print("PYTHONPATH New:", sys.path)

    # Synchronize all GPUs to wait until rank 0 completes the above commands
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Deformable DETR Detector for implants', parents=[get_args_parser()])
    args = parser.parse_args()

    compile_kernels()

    command = f"python models/main_azure.py " + " ".join(f"--{k} {v}" for k, v in vars(args).items() if v is not None)
    # print(f"\nExecuting command: '{command}'")

    os.system(command)
