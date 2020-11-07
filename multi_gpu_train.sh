#! /usr/bin/env bash

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=1,0 \
python -m torch.distributed.launch \
--nproc_per_node=2 \
--nnodes=1 \
--node_rank=0 \
main/train.py -c configs/resnet50_train.yaml

pkill -f torch/bin/python  # Kill previous died processes