#!/bin/bash

#SBATCH -N 1
#SBATCH -M swarm
#SBATCH -p gpu
#SBATCH -n 20
#SBATCH --gres=gpu:4

#python -m torch.distributed.launch \
#   --nproc_per_node=4 \
#   train_fp16.py --config-file $1
#python train.py --config-file $1 $2
python train.py --config-file $1 --gpu-id $2 $3
