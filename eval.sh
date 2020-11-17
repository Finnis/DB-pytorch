#! /usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 \
python main/eval.py -c configs/resnet50_eval_pred.yaml