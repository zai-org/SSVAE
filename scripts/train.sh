#! /bin/bash

config_path=$1
exp_name=$2

torchrun --nproc-per-node 8 main.py --base $config_path --seed 0 --name $exp_name --wandb
