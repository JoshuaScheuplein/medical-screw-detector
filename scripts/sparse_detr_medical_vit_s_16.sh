#!/bin/bash -l
#SBATCH --job-name=DAX-Screw-Detection
#SBATCH --output=logfile_output_%j.log
#SBATCH --error=logfile_error_%j.log
#SBATCH --time=0-23:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --mail-user=joshua.scheuplein.ext@siemens-healthineers.com
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --export=NONE         # Do not export environment from submitting shell
unset SLURM_EXPORT_ENV        # Enable export of environment from this script to srun

############  DESCRIPTION  #################

# Screw Detection with sparse DETR and CIA layer

# Checkpoint: DINO_Training_Job_037_ViT-S-16_0200

############  FILE PATHS  ##################

CHECKPOINT="$HPCVAULT/DINO-Checkpoints/checkpoint_vit_small_DINO_Training_Job_037_ViT-S-16_0200.pth"

SRC_DIR="$HOME/medical-screw-detector"

RESULTS_DIR="$HOME/Screw-Detection-Results/Job-$SLURM_JOB_ID"

FAST_DATA_DIR="$TMPDIR/Job-$SLURM_JOB_ID"

DATA_ARCHIVE="$HPCVAULT/V1-1to3objects-400projections-circular.tar.gz"

##############   TRAINING   ###############

# Load and init conda if it doesn't exist
if ! command -v conda &> /dev/null
then
echo "load conda from module"
module load python/3.10-anaconda
conda init bash
fi

# Activate conda environment
source activate DETRenv
echo -e "\nUsing Python: $(which python)\n"

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Check that python can access GPU instance
nvidia-smi
echo -e "\nPython reaches GPU: $(python -c 'import torch; print(torch.cuda.is_available())')\n"

# Copy bash script for reproducibility
mkdir -p "$RESULTS_DIR"
cd "$SRC_DIR/scripts" || echo "Error: Failed to change into $SRC_DIR/scripts"
cp "$SRC_DIR/scripts/sparse_detr_medical_vit_s_16.sh" "$RESULTS_DIR/sparse_detr_medical_vit_s_16_$SLURM_JOB_ID.sh"

# Copy training data to faster drive
mkdir -p "$FAST_DATA_DIR"
echo "Successfully created $FAST_DATA_DIR"

echo -e "\nStarted data transfer at $(date)"
tar -xf "$DATA_ARCHIVE" -C "$FAST_DATA_DIR"
echo -e "\nFinished data transfer at $(date)\n"

# Display disk usage of $FAST_DATA_DIR
cd "$FAST_DATA_DIR" || echo "Error: Failed to change into $FAST_DATA_DIR"
du -ah -d 1

# Display general disk usage
shownicerquota.pl

# Change to source directory and retrieve infos about most recent git commit
cd "$SRC_DIR" || echo "Error: Failed to change into $SRC_DIR"
echo -e "\nUsing Repository Revision:"
git log --oneline -n 1

# Start model training
echo -e "\nTraining started at $(date)"

# Note: Dafault batch size = 3
srun python3 main.py \
  --job_ID "$SLURM_JOB_ID" \
  --data_dir "$FAST_DATA_DIR/V1-1to3objects-400projections-circular" \
  --result_dir "$RESULTS_DIR" \
  --backbone "medical_vit_s_16" \
  --backbone_checkpoint_file "$CHECKPOINT" \
  --dataset_reduction 2 \
  --log_wandb \
  --lr 0.00004 \
  --lr_drop_epochs 40 \
  --lr_backbone 0.000004 \
  --batch_size 6 \
  --epochs 50 \
  --with_box_refine \
  --two_stage \
  --eff_query_init \
  --eff_specific_head \
  --rho 0.1 \
  --use_enc_aux_loss \
  --num_queries 300 \
  --num_workers 10

# save predictions to `HOME`
# cd "$TMPDIR" && rsync -avRP --relative --include='*/' --include='predictions*.json' --exclude='*' "./V1-1to3objects-400projections-circular/" $result_dir

echo -e "\nTraining completed at $(date)"

# Clean up temporary directory
rm -rf "$FAST_DATA_DIR"
echo -e "\nSuccessfully cleaned up $FAST_DATA_DIR"

echo -e "\nHPC-Cluster job $SLURM_JOB_ID successfully finished! :)\n"
