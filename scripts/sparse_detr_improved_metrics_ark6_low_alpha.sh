#!/bin/bash -l
#SBATCH --job-name=sparse_detr_ark6_comparison_v2
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=0-23:30:00  # Set the time limit for the job
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --export=NONE
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
unset SLURM_EXPORT_ENV

module load python/3.10-anaconda
export PYTHONPATH=/home/hpc/iwi5/iwi5163h/dinov2_playground

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# TODO:
# conda activate YOUR_ENVIRONMENT

current_datetime=$(date +"%Y_%m_%d_%H_%M")
result_dir="$HOME/results/sparse_detr_improved_metrics_ark6_low_alpha"

# copy data to `$TMPDIR`
cd "$HPCVAULT" && rsync -aRP --relative --update --include="*/" --include='encodings.tiff' --include='projections.tiff' --include='labels.json' --exclude="*/**" ./V1-1to3objects-400projections-circular/ "$TMPDIR"

srun python3 ~/dinov2_playground/improved_detr/main.py \
  --data_dir "$TMPDIR/V1-1to3objects-400projections-circular"  \
  --backbone "ark6_low_res" \
  --dataset_reduction 2 \
  --log_wandb \
  --lr 0.00004 \
  --lr_backbone 0.000004 \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop_epochs 40 \
  --with_box_refine \
  --two_stage \
  --eff_query_init \
  --eff_specific_head \
  --rho 0.1 \
  --use_enc_aux_loss \
  --num_queries 300 \
  --result_dir $result_dir \
  --alpha_correspondence

# save predictions to `HOME`
cd "$TMPDIR" && rsync -avRP --relative --include='*/' --include='predictions*.json' --exclude='*' "./V1-1to3objects-400projections-circular/" $result_dir