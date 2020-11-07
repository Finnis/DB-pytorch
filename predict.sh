#! /usr/bin/env bash

rm -rf results/*

CUDA_VISIBLE_DEVICES=1 \
python main/predict.py \
-c configs/resnet18_eval_pred.yaml \
--save_poly \
--save_prob_map \
--show_poly \
--image_path=datasets/Container/test_images