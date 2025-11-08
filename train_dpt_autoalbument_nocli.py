import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import DataLoader
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from dataset.dataset import MultiTaskDataSet
import albumentations as A

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
    print("Loaded AutoAlbument policy:", policy)
    
    # Create a list of transforms
    transforms = [
        A.RandomCrop(crop_size, crop_size),
        policy,
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)

def get_dataset_config():
    """Return dataset configuration."""
    return {
        'root_dir': '/mnt/synrs3d/SynRS3D/data',
        'datasets': ['grid_g05_mid_v2', 'grid_g005_mid_v2', 'terrain_g05_low_v1', 'terrain_g005_low_v1', 
                    'grid_g05_mid_v1', 'grid_g005_mid_v1', 'terrain_g05_mid_v1', 'terrain_g005_mid_v1', 
                    'grid_g05_low_v1', 'grid_g005_low_v1', 'grid_g05_high_v1', 'grid_g005_high_v1', 
                    'terrain_g05_high_v1', 'terrain_g005_high_v1', 'terrain_g1_low_v1', 'terrain_g1_mid_v1', 
                    'terrain_g1_high_v1'],
        'tgt_datasets': [],  # Add target datasets if needed for domain adaptation
        'images_file': ['train.txt', 'test_syn.txt', 'test.txt'],
        'crop_size': 392,
        'is_training': True,
        'multi_task': True,
        'combine_class': True,
        'max_iters': None,
        'max_da_images': 500000,
        'apply_da': [],  # Add data augmentation methods if needed
        'da_aug_paras': None  # Add augmentation parameters if needed
    }

def visualize_samples(dataset, num_samples=3):
    """Visualize original and transformed images and masks."""
    # Create a figure with 4 columns: original image, transformed image, original mask, transformed mask
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array for consistent indexing
    
    # Get first few samples
    num_samples = min(num_samples, len(dataset))  # Ensure we don't exceed dataset size
    
    # Get a sample to check available keys
    sample = dataset[0]
    has_mask = 'mask' in sample
    
    for i in range(num_samples):
        # Get original sample
        sample = dataset.get_original_sample(i) if hasattr(dataset, 'get_original_sample') else dataset[i]
        orig_image = sample['image']
        orig_mask = sample.get('mask')
        
        # Get transformed sample
        sample = dataset[i]
        trans_image = sample['image']
        trans_mask = sample.get('mask')
        
        def prepare_image(img):
            """Convert and normalize image for display."""
            if torch.is_tensor(img):
                # Convert to numpy and ensure channel dimension is last
                if img.dim() == 3 and img.shape[0] in [1, 3]:  # C, H, W format
                    img = img.permute(1, 2, 0)  # Convert to H, W, C
                img = img.numpy()
            
            # Handle single-channel images by repeating the channel
            if len(img.shape) == 2:  # H, W
                img = np.stack([img] * 3, axis=-1)  # Convert to H, W, 3
            
            # Ensure float32 and normalize to [0, 1]
            img = img.astype(np.float32)
            if img.max() > 1.0 or img.min() < 0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return np.clip(img, 0, 1)
        
        def prepare_mask(mask_img):
            """Convert and prepare mask for display."""
            if mask_img is None:
                return None
                
            if torch.is_tensor(mask_img):
                mask_img = mask_img.squeeze()
                if mask_img.dim() > 2:  # If there's a channel dimension
                    mask_img = mask_img.permute(1, 2, 0).squeeze()
                mask_img = mask_img.numpy()
            
            # For segmentation mask, ensure it's integer type
            mask_img = mask_img.astype(np.int32)
            
            # Create a colored mask using a colormap
            unique_vals = np.unique(mask_img)
            num_classes = len(unique_vals)
            cmap = plt.cm.get_cmap('tab20', max(num_classes, 20))  # Use at least 20 colors for better visibility
            
            # Normalize to [0, 1] for colormap
            if num_classes > 1:
                norm_mask = (mask_img - unique_vals.min()) / (unique_vals.max() - unique_vals.min() + 1e-8)
            else:
                norm_mask = mask_img.astype(float)
                
            colored_mask = cmap(norm_mask)
            return colored_mask, num_classes
        
        # Plot original image
        orig_img_disp = prepare_image(orig_image)
        axes[i, 0].imshow(orig_img_disp)
        axes[i, 0].set_title(f'Sample {i+1}: Original Image')
        axes[i, 0].axis('off')
        
        # Plot transformed image
        trans_img_disp = prepare_image(trans_image)
        axes[i, 1].imshow(trans_img_disp)
        axes[i, 1].set_title('Transformed Image')
        axes[i, 1].axis('off')
        
        # Plot original mask if available
        if orig_mask is not None:
            orig_mask_disp, num_classes = prepare_mask(orig_mask)
            axes[i, 2].imshow(orig_mask_disp)
            axes[i, 2].set_title(f'Original Mask ({num_classes} classes)')
            axes[i, 2].axis('off')
        
        # Plot transformed mask if available
        if trans_mask is not None:
            trans_mask_disp, num_classes = prepare_mask(trans_mask)
            axes[i, 3].imshow(trans_mask_disp)
            axes[i, 3].set_title(f'Transformed Mask ({num_classes} classes)')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Get dataset configuration
    config = get_dataset_config()
    
    # Prepare data paths
    root_dir = config['root_dir']
    syn_train_data_path = [os.path.join(root_dir, dataset) for dataset in config['datasets']]
    tgt_data_path = [os.path.join(root_dir, dataset) for dataset in config['tgt_datasets']]
    
    # Initialize transforms
    print("Initializing transforms...")
    policy_path = "/path/to/your/policy.json"  # Update this path
    training_src_transforms = get_autoalbument_transforms(policy_path, crop_size=config['crop_size'])
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = MultiTaskDataSet(
        root=syn_train_data_path,
        is_training=config['is_training'],
        images_file=config['images_file'],
        transforms=training_src_transforms,
        max_iters=config['max_iters'],
        max_da_images=config['max_da_images'],
        multi_task=config['multi_task'],
        combine_class=config['combine_class'],
        apply_da=config['apply_da'],
        da_aug_paras=config['da_aug_paras'],
        tgt_root_dir=tgt_data_path
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize samples
    print("Visualizing sample images...")
    visualize_samples(dataset, num_samples=3)
