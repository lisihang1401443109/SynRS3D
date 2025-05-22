#!/bin/bash

# Check if the required argument (stylized_p) is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <stylized_p>"
    exit 1
fi

stylized_p=$1  # Take the first argument as stylized_p

train_set=('grid_g05_mid_v2' 'grid_g005_mid_v2' 'terrain_g05_low_v1' 'terrain_g005_low_v1' 'grid_g05_mid_v1' 'grid_g005_mid_v1' 'terrain_g05_mid_v1' 'terrain_g005_mid_v1' 'grid_g05_low_v1' 'grid_g005_low_v1' 'grid_g05_high_v1' 'grid_g005_high_v1' 'terrain_g05_high_v1' 'terrain_g005_high_v1' 'terrain_g1_low_v1' 'terrain_g1_mid_v1' 'terrain_g1_high_v1')

test_set=('DFC18' 'DFC19_JAX' 'DFC19_OMA' 'geonrw_rural' 'geonrw_urban' 'OGC_ARG' 'OGC_ATL')

images_file=('train.txt' 'test_syn.txt' 'train.txt')

# Update snapshot directory name
snapshot_dir="/mnt/synrs3d/SynRS3D/stylized_src_only_${stylized_p}_100k"

python train_dpt_sourceonly.py \
--root_dir /mnt/synrs3d/SynRS3D/data \
--datasets ${train_set[*]} \
--test_datasets ${test_set[*]} \
--ood_datasets ${test_set[*]} \
--crop_size 392 \
--encoder vitl \
--decoder DPT \
--snapshot_dir ${snapshot_dir} \
--images_file ${images_file[*]} \
--batch_size 1 \
--learning_rate 1e-6 \
--weight_decay 5e-4 \
--warmup_steps 0 \
--decay_mode poly \
--num_steps 100000 \
--save_num_images 0 \
--save_pred_every 1000 \
--multi_task \
--pretrained \
--shuffle \
--only_save_best \
--decoder_lr_weight 10 \
--lambda_dsms 1.0 \
--stylized_p ${stylized_p} \
--eval_oem
#optional
# --eval_oem
