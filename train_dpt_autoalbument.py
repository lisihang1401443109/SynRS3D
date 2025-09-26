import os
import sys
import argparse
import os.path as osp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
import json
import numpy as np
from evaluation import eval, eval_oem
from evaluation import custom_collate
from utils.utils import adjust_learning_rate
from utils.datasets_config import (
    ss_datasetname, dataset_num_classes, get_dataset_category
)
from torch.utils import data
from ever.core.logger import get_console_file_logger
from dataset.dataset import MultiTaskDataSet, OEMDataSet
from utils.criterion import SmoothL1Loss, CriterionCrossEntropy
from models.dpt import DPT_DINOv2
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RS3DAda with AutoAlbument")

    parser.add_argument("--root_dir", type=str, default='/home/songjian/project/SynRS3D/data/', 
                       help="Path to the directory containing the datasets.")
    parser.add_argument("--datasets", nargs='*', type=str, default=['grid_g05_mid_v1'], 
                       help="training datasets name list")
    parser.add_argument("--test_datasets", nargs='*', type=str, default=['DFC18'], 
                       help="target domain 1 datasets list and target domain 2 datasets list")
    parser.add_argument("--ood_datasets", nargs='*', type=str, default=['DFC18'], 
                       help="target domain 2 datasets list")
    parser.add_argument("--images_file", nargs='*', type=str, default=['train.txt', 'test.txt', 'train.txt'], 
                       help="images txt file, first one is the training txt, second is the test txt, third is the style transfer txt")
    parser.add_argument("--crop_size", type=int, default=392, 
                       help="height and width of images.")
    parser.add_argument('--decoder', type=str, default='DPT',
                       help='decoder')
    parser.add_argument('--encoder', type=str, default='vitl',
                       help='encoder')
    parser.add_argument("--multi_task", action="store_true", 
                       help="Whether to add segmentation branch.")
    parser.add_argument("--combine_class", action="store_true", 
                       help="Whether to combine 8 classes to 3.")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="batchsize")
    parser.add_argument("--learning_rate", type=float, default=1e-6, 
                       help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--decoder_lr_weight", type=float, default=10, 
                       help="weight of decoder lr, default are 10 times of encoder's lr")
    parser.add_argument("--num_steps", type=int, default=40000, 
                       help="Number of training steps.")
    parser.add_argument("--start_iters", type=int, default=0, 
                       help="start_iters")
    parser.add_argument("--power", type=float, default=0.9, 
                       help="Decay parameter to compute the learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=1500, 
                       help="Number of warm-up steps.")
    parser.add_argument("--warmup_mode", type=str, default='linear', 
                       help="warm-up mode")
    parser.add_argument("--decay_mode", type=str, default='poly', 
                       help="decay mode")
    parser.add_argument("--weight_decay", type=float, default=5e-4, 
                       help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='0', 
                       help="choose gpu device.")
    parser.add_argument("--save_num_images", type=int, default=5, 
                       help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=500, 
                       help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot_dir", type=str, default='snapshot_autoalbument', 
                       help="Where to save snapshots of the model.")
    parser.add_argument("--only_save_best", action="store_true", 
                       help="only save best checkpoint")
    parser.add_argument("--lambda_dsms", type=float, default=0.8, 
                       help="weight of height estimation loss")
    parser.add_argument("--eval_oem", action="store_true", 
                       help="evaluation on OEM dataset or not")
    parser.add_argument("--pretrained", action="store_true", 
                       help="use pretrained DINOv2 or not.")
    parser.add_argument("--shuffle", action="store_true", 
                       help="shuffle or not")
    parser.add_argument("--feat_loss", action="store_true", 
                       help="use feature constraint loss or not")
    parser.add_argument("--fl_start", type=int, default=3, 
                       help="calculate feature loss from which layer")
    parser.add_argument("--fl_threshold", type=float, default=0.8, 
                       help="threshold, ϵ in formula [4]")
    parser.add_argument("--fl_weight", type=float, default=1., 
                       help="weight of feature constraint loss")
    parser.add_argument("--fl_decrement", type=float, default=0.05, 
                       help="This value determines how much the threshold decreases per layer")
    
    # AutoAlbument specific arguments
    parser.add_argument("--policy_path", type=str, 
                       default='/mnt/synrs3d/SynRS3D/autoalbument/outputs/2025-09-09/01-31-02/policy/latest.json',
                       help="Path to the AutoAlbument policy JSON file.")

    return parser.parse_args()

def get_autoalbument_transforms(policy_path, crop_size=392):
    """Load and configure AutoAlbument policy.
    
    Args:
        policy_path: Path to the AutoAlbument policy JSON file
        crop_size: Size for random crop
        
    Returns:
        A composed Albumentations transform
    """
    # Load the AutoAlbument policy
    policy = A.load(policy_path, data_format='json')
    
    # Create a list of transforms
    transforms = [
        A.RandomCrop(crop_size, crop_size, always_apply=True),
        policy,
        # ToTensorV2()
    ]
    
    return A.Compose(transforms)

def main():
    args = get_arguments()
    
    # Setup logging and model directory
    snapshot_dir = os.path.join(args.snapshot_dir, f"{args.decoder}_{args.encoder}", 
                               f"{args.crop_size}_lr_{args.learning_rate}_wd_{args.weight_decay}")
    
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    
    # Save config
    with open(os.path.join(snapshot_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    logger = get_console_file_logger(
        name=f"{args.decoder}_{args.encoder}_autoalbument", 
        level=logging.INFO, 
        logdir=snapshot_dir
    )
    
    writer = SummaryWriter(log_dir=os.path.join(snapshot_dir, 'runs'))
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup model
    ss_num_classes = 3 if args.combine_class else dataset_num_classes[get_dataset_category(set(args.datasets))]
    
    regression_config = [{'name': 'regression', 'nclass': 1}]
    segmentation_config = [{'name': 'segmentation', 'nclass': ss_num_classes}]
    
    head_configs = regression_config + segmentation_config if args.multi_task else regression_config
    
    model = DPT_DINOv2(
        encoder=args.encoder, 
        head_configs=head_configs, 
        pretrained=args.pretrained  # Use pretrained weights if specified
    ).to(device)
    
    # Setup data loaders with AutoAlbument transforms
    train_transforms = get_autoalbument_transforms(args.policy_path, args.crop_size)
    
    # Training dataset
    train_data_path = [os.path.join(args.root_dir, dataset) for dataset in args.datasets]
    train_dataset = MultiTaskDataSet(
        train_data_path,
        is_training=True,
        images_file=args.images_file,
        transforms=train_transforms,
        max_iters=args.num_steps * args.batch_size,
        multi_task=args.multi_task,
        combine_class=args.combine_class
    )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Validation dataset (no data augmentation)
    val_transforms = A.Compose([
        A.CenterCrop(504, 504, always_apply=True),
        A.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            max_pixel_value=255.0,
            always_apply=True
        ),
        ToTensorV2()
    ])
    
    val_loaders = {}
    for dataset_name in args.test_datasets:
        val_dataset = MultiTaskDataSet(
            [os.path.join(args.root_dir, dataset_name)],
            is_training=False,
            images_file=args.images_file,
            transforms=val_transforms,
            multi_task=args.multi_task,
            combine_class=args.combine_class
        )
        
        val_loaders[dataset_name] = data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    # Setup optimizer and loss functions
    optimizer = optim.AdamW(
        [
            {'params': model.pretrained.parameters(), 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() 
                       if 'pretrained' not in n], 'lr': args.learning_rate * args.decoder_lr_weight}
        ],
        weight_decay=args.weight_decay
    )
    
    height_criterion = SmoothL1Loss(reduction='mean')
    if args.multi_task:
        segmentation_criterion = CriterionCrossEntropy(ignore_index=255)
    
    # Training loop
    model.train()
    global_step = 0
    best_metric = float('inf')  # Assuming lower is better (e.g., RMSE)
    
    while global_step < args.num_steps:
        for batch in train_loader:
            if global_step >= args.num_steps:
                break
                
            # Get batch data
            images = batch['image'].to(device)
            height_maps = batch['height'].to(device)
            
            if args.multi_task:
                seg_maps = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate losses
            height_pred = outputs['regression']
            height_loss = height_criterion(height_pred, height_maps.unsqueeze(1))
            
            if args.multi_task:
                seg_pred = outputs['segmentation']
                seg_loss = segmentation_criterion(seg_pred, seg_maps.long())
                loss = height_loss + seg_loss
            else:
                loss = height_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if global_step % 100 == 0:
                log_str = f"Step [{global_step}/{args.num_steps}] Loss: {loss.item():.4f}"
                if args.multi_task:
                    log_str += f" | Height Loss: {height_loss.item():.4f}, Seg Loss: {seg_loss.item():.4f}"
                logger.info(log_str)
                
                # Log to TensorBoard
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/height_loss', height_loss.item(), global_step)
                if args.multi_task:
                    writer.add_scalar('train/seg_loss', seg_loss.item(), global_step)
            
            # Validation and model saving
            if (global_step + 1) % args.save_pred_every == 0 or (global_step + 1) == args.num_steps:
                model.eval()
                val_metrics = {}
                
                with torch.no_grad():
                    for dataset_name, val_loader in val_loaders.items():
                        val_loss = 0.0
                        val_height_loss = 0.0
                        val_seg_loss = 0.0 if args.multi_task else None
                        
                        for val_batch in val_loader:
                            val_images = val_batch['image'].to(device)
                            val_height_maps = val_batch['height'].to(device)
                            
                            val_outputs = model(val_images)
                            
                            # Calculate validation losses
                            val_height_pred = val_outputs['regression']
                            val_height_loss = height_criterion(val_height_pred, val_height_maps.unsqueeze(1))
                            val_height_loss += val_height_loss.item()
                            
                            if args.multi_task:
                                val_seg_maps = val_batch['mask'].to(device)
                                val_seg_pred = val_outputs['segmentation']
                                val_seg_loss += segmentation_criterion(val_seg_pred, val_seg_maps.long()).item()
                        
                        # Average losses
                        val_height_loss /= len(val_loader)
                        val_loss = val_height_loss
                        
                        if args.multi_task:
                            val_seg_loss /= len(val_loader)
                            val_loss += val_seg_loss
                        
                        # Log validation metrics
                        val_metrics[dataset_name] = {
                            'loss': val_loss,
                            'height_loss': val_height_loss,
                        }
                        if args.multi_task:
                            val_metrics[dataset_name]['seg_loss'] = val_seg_loss
                        
                        # Log to console
                        log_str = f"Validation - {dataset_name} | Loss: {val_loss:.4f} | Height Loss: {val_height_loss:.4f}"
                        if args.multi_task:
                            log_str += f" | Seg Loss: {val_seg_loss:.4f}"
                        logger.info(log_str)
                        
                        # Log to TensorBoard
                        writer.add_scalar(f'val/{dataset_name}/loss', val_loss, global_step)
                        writer.add_scalar(f'val/{dataset_name}/height_loss', val_height_loss, global_step)
                        if args.multi_task:
                            writer.add_scalar(f'val/{dataset_name}/seg_loss', val_seg_loss, global_step)
                
                # Save model if it's the best so far
                # Here we use the first validation dataset as the main metric for model selection
                main_val_metric = list(val_metrics.values())[0]['height_loss']
                if main_val_metric < best_metric:
                    best_metric = main_val_metric
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics,
                    }, os.path.join(snapshot_dir, 'best_model.pth'))
                    logger.info(f"Saved new best model at step {global_step} with val height loss: {main_val_metric:.4f}")
                
                # Always save the latest model
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, os.path.join(snapshot_dir, 'latest_model.pth'))
                
                model.train()
            
            global_step += 1
    
    logger.info("Training completed!")
    writer.close()

if __name__ == '__main__':
    main()
