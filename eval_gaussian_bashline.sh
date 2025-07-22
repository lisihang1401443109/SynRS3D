#!/bin/bash

test_set=('DFC18' 'DFC19_JAX' 'DFC19_OMA' 'geonrw_rural' 'geonrw_urban' 'OGC_ARG' 'OGC_ATL')

python evaluation.py \
--restore_path /mnt/synrs3d/SynRS3D/snapshot_so_gaussian_50k/DPT_vitl/392_lr_1e-06_wd_0.0005/multi_task_ori_class/best_HE_model_11500.pth \
--test_datasets ${test_set[*]}  \
--ood_datasets ${test_set[*]} \
--pretrained \
--gpu 0 \
--images_file train.txt test.txt test.txt \
--snapshot_dir snapshot_so_gaussian_50k_eval \
--root_dir /mnt/synrs3d/SynRS3D/data \

# Note: OEM evaluation is not included as requested
