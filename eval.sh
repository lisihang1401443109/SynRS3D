#/bin/bash
python evaluation.py \
--restore_path /mnt/synrs3d/SynRS3D/pretrain/RS3DAda_vitl_DPT_segmentation.pth \
--test_datasets OEM \
--ood_datasets DFC19_OMA \
--pretrained \
--gpu 0 \
--images_file train.txt test.txt test.txt \
--snapshot_dir snapshot_oem \
--combine_class \
--eval_oem
