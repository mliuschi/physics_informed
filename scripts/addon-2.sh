#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re500to500-1s.yaml \
--start 0 \
--stop 40 \
--log;
CUDA_VISIBLE_DEVICES=0 python3 run_pino3d.py \
--config_path configs/transfer/Re250to300-1s.yaml \
--start 33 \
--stop 40 \
--log;


