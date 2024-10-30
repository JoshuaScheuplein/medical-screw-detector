######################################################################
# The following packages are just imported to test the environment ...
######################################################################
import os
import time
import argparse
import signal
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback
######################################################################

print("\n#################################################################'")
print("#################################################################'")
print("\nSuccessfully imported necessary packages for SparseDETR!")

import os
import subprocess

import torch

cuda_available = torch.cuda.is_available()
print("\nCUDA available:", cuda_available)
if cuda_available:
    print("Current GPU:", torch.cuda.get_device_name(0)) # '0' is the index of the first GPU
else:
    print("No GPU available!")

try:
    import MultiScaleDeformableAttention
except Exception as e:
    print("\nCould not import package 'MultiScaleDeformableAttention' ...")
    print("Exception:", e)

print("\nCurrent directory:", os.getcwd())
print("Files in the current directory:", os.listdir("."))

os.system("echo '\nExecuting <ls> command:'")
os.system("ls")

os.system("echo '\nExecuting <pwd> command:'")
os.system("pwd")

os.system("echo '\n#################################################################'")
os.system("echo 'Executing <setup.py>:'")
os.system("echo '#################################################################'")
os.system("python setup.py build install") # Compile CUDA kernels
os.system("echo '#################################################################'")

# os.system("chmod +x make.sh")
# os.system("./make.sh")

# subprocess.run(["sh", "chmod +x make.sh"], check=True)
# subprocess.run(["sh", "./make.sh"], check=True)

os.system("echo '\n#################################################################'")
os.system("echo 'Available conda environments:'")
os.system("echo '#################################################################'")
os.system("conda env list")

os.system("echo '\n#################################################################'")
os.system("echo 'Available packages in active environment:'")
os.system("echo '#################################################################'")
os.system("conda list")

os.system("echo '\n#################################################################'")
os.system("echo 'Running unit test script <test.py>:'")
os.system("echo '#################################################################'")
os.system("python test.py")
os.system("echo '#################################################################'")

os.system("echo '\n#################################################################'")
os.system("echo 'Test import of built package <MultiScaleDeformableAttention>:'")
os.system("echo '#################################################################'")
os.system("python functions/test_import.py")
os.system("echo '#################################################################'")
