#/bin/bash
python evaluation.py \
--restore_path /mnt/synrs3d/SynRS3D/snapshot_src_only/DPT_vitl/392_lr_1e-06_wd_0.0005/multi_task_ori_class/best_SS_model_25500.pth \
--test_datasets OEM \
--ood_datasets DFC19_OMA \
--pretrained \
--gpu 0 \
--images_file train.txt test.txt test.txt \
--snapshot_dir snapshot_oem \
--eval_oem
