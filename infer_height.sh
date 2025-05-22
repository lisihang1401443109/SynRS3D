#!bin/bash

python infer_height.py \
--data_dir ./data \
--restore_from ./pretrain/RS3DAda_vitl_DPT_height.pth \
--output_path  ./output/jan24 \
--use_tta