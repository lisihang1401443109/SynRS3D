#!/bin/bash

# Training and test datasets
train_set=('grid_g05_mid_v2' 'grid_g005_mid_v2' 'terrain_g05_low_v1' 'terrain_g005_low_v1' 'grid_g05_mid_v1' 'grid_g005_mid_v1' 'terrain_g05_mid_v1' 'terrain_g005_mid_v1' 'grid_g05_low_v1' 'grid_g005_low_v1' 'grid_g05_high_v1' 'grid_g005_high_v1' 'terrain_g05_high_v1' 'terrain_g005_high_v1' 'terrain_g1_low_v1' 'terrain_g1_mid_v1' 'terrain_g1_high_v1')
test_set=('DFC18' 'DFC19_JAX' 'DFC19_OMA' 'geonrw_rural' 'geonrw_urban' 'OGC_ARG' 'OGC_ATM')

# Path to the AutoAlbument policy
POLICY_PATH="/mnt/synrs3d/SynRS3D/autoalbument/outputs/2025-09-09/01-31-02/policy/latest_compatible.json"

# Training parameters
python train_dpt_autoalbument.py \
--root_dir /mnt/synrs3d/SynRS3D/data \
--datasets ${train_set[*]} \
--test_datasets ${test_set[*]} \
--crop_size 392 \
--encoder vitl \
--decoder DPT \
--snapshot_dir /mnt/synrs3d/SynRS3D/snapshot_autoalbument_50k \
--policy_path $POLICY_PATH \
--batch_size 2 \
--learning_rate 1e-6 \
--weight_decay 5e-4 \
--num_steps 50000 \
--save_pred_every 500 \
--multi_task \
--pretrained \
--combine_class \
--decoder_lr_weight 10 \
--gpu 0,1  # Adjust based on available GPUs

# Optional: Uncomment to evaluate on OEM dataset
# --eval_oem
