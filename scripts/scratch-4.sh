#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/scratch/Re400-scratch-1s.yaml \
--start 40 \
--stop 80 \
--log