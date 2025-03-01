#!/bin/bash
#SBATCH --job-name=surgical-finetune
#SBATCH --gres=gpu:2  # Fewer GPUs needed
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

srun python finetune.py \
  --pretrained_path /path/to/pretrained_checkpoint.pth \
  --annotation_path /path/to/finetune_annotations.csv \
  --video_root /path/to/finetune_data \
  --epochs 50