#!/bin/bash
#SBATCH --job-name=surgical-pretrain
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --partition=gpu

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

srun python main.py \
  --annotation_path /path/to/annotations.csv \
  --video_root /path/to/video_frames \
  --batch_size 256 \
  --epochs 100 \
  --world_size 8 \
  --save_dir /path/to/checkpoints