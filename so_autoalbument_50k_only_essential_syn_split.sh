#!/bin/bash
POLICY_PATH="${1:-}"
SNAPSHOT_DIR="${2:-/mnt/synrs3d/SynRS3D/snapshot_so_autoalbument_50k}"
if [ -z "$POLICY_PATH" ]; then
  echo "Usage: $0 <POLICY_PATH> [SNAPSHOT_DIR]"
  exit 1
fi
train_set=('grid_g05_mid_v2' 'grid_g005_mid_v2' 'terrain_g05_low_v1' 'terrain_g005_low_v1' 'grid_g05_mid_v1' 'grid_g005_mid_v1' 'terrain_g05_mid_v1' 'terrain_g005_mid_v1' 'grid_g05_low_v1' 'grid_g005_low_v1' 'grid_g05_high_v1' 'grid_g005_high_v1' 'terrain_g05_high_v1' 'terrain_g005_high_v1' 'terrain_g1_low_v1' 'terrain_g1_mid_v1' 'terrain_g1_high_v1')
# train_set=('grid_g005_mid_v2')
# test_set=('DFC18' 'DFC19_JAX' 'DFC19_OMA' 'geonrw_rural' 'geonrw_urban' 'OGC_ARG' 'OGC_ATL')
test_set=('grid_g05_mid_v2' 'grid_g005_mid_v2' 'terrain_g05_low_v1' 'terrain_g005_low_v1' 'grid_g05_mid_v1' 'grid_g005_mid_v1' 'terrain_g05_mid_v1' 'terrain_g005_mid_v1' 'grid_g05_low_v1' 'grid_g005_low_v1' 'grid_g05_high_v1' 'grid_g005_high_v1' 'terrain_g05_high_v1' 'terrain_g005_high_v1' 'terrain_g1_low_v1' 'terrain_g1_mid_v1' 'terrain_g1_high_v1')

# images_file=('train.txt' 'test_syn.txt' 'train.txt')
images_file=('train_syn.txt' 'test_syn_2.txt' 'train.txt')
da=()

# POLICY_PATH="/mnt/synrs3d/SynRS3D/autoalbument/outputs/2025-09-09/01-31-02/policy/latest_392.json"
# POLICY_PATH="/mnt/synrs3d/SynRS3D/autoalbument/outputs/2025-10-05/17-13-57/policy/latest_392.json"

python train_dpt_autoalbument_only_essential.py \
--root_dir /mnt/synrs3d/SynRS3D/data \
--datasets ${train_set[*]} \
--test_datasets ${test_set[*]} \
--ood_datasets ${test_set[*]} \
--crop_size 392 \
--encoder vitl \
--decoder DPT \
--snapshot_dir "$SNAPSHOT_DIR" \
--images_file ${images_file[*]} \
--batch_size 8 \
--learning_rate 1e-6 \
--weight_decay 5e-4 \
--warmup_steps 0 \
--decay_mode poly \
--num_steps 50000 \
--save_num_images 0 \
--save_pred_every 1000 \
--multi_task \
--pretrained \
--shuffle \
--only_save_best \
--decoder_lr_weight 10 \
--policy_path "$POLICY_PATH" \
--lambda_dsms 1.0 \
#optional
#--eval_oem