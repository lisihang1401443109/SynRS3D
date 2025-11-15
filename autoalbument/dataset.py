import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any, Tuple
import random

# Disable OpenCL and multithreading for consistency
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class AutoAlbumentDataset(Dataset):
    """
    A dataset class for AutoAlbument that returns (image, mask) pairs with one-hot encoded masks.
    """
    
    def __init__(
        self,
        root: str,
        image_dir: str = "opt_orig",
        mask_dir: str = "gt_ss_mask",
        split_file: str = "train.txt",
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
        num_classes: int = 5,
    ):
        """
        Args:
            root: Root directory of the dataset
            image_dir: Subdirectory containing the images
            mask_dir: Subdirectory containing the segmentation masks
            split_file: Text file containing list of image filenames (without extensions)
            transform: Albumentations transform to be applied
            max_samples: Maximum number of samples to use (for debugging)
            num_classes: Number of classes in the segmentation task
        """
        self.root = root
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.transform = transform
        self.num_classes = num_classes
        
        # Read image IDs from split file
        with open(os.path.join(root, split_file), 'r') as f:
            self.image_ids = [line.strip() for line in f]
            
        if max_samples is not None:
            self.image_ids = self.image_ids[:max_samples]
        
        # Store image and mask paths for faster access
        self.image_paths = [os.path.join(self.image_dir, f"{img_id}.tif") for img_id in self.image_ids]
        self.mask_paths = [os.path.join(self.mask_dir, f"{img_id}.tif") for img_id in self.image_ids]
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _convert_to_one_hot(self, mask: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to one-hot encoding."""
        height, width = mask.shape[:2]
        one_hot = np.zeros((height, width, self.num_classes), dtype=np.float32)
        
        for class_id in range(self.num_classes):
            one_hot[:, :, class_id] = (mask == class_id).astype(float)
            
        return one_hot
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            Tuple containing:
                - image: numpy array of shape (H, W, 3)
                - mask: one-hot encoded mask of shape (H, W, num_classes)
        """
        # Load image and mask using OpenCV for consistency
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Load mask and ensure it's 2D
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]  # Take first channel if mask is multi-channel
        
        # Convert to one-hot encoding
        mask_one_hot = self._convert_to_one_hot(mask)
        
        # Apply transforms if specified
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask_one_hot)
            image = transformed['image']
            mask_one_hot = transformed['mask']
        print(image.shape, mask_one_hot.shape)
        
        return image, mask_one_hot

def get_dataset(
    data_dir: str,
    image_dir: str = "opt_orig",
    mask_dir: str = "gt_ss_mask",
    split: str = "train",
    transform: Optional[Any] = None,
    max_samples: Optional[int] = None,
    num_classes: int = 5,
) -> AutoAlbumentDataset:
    """
    Helper function to create a dataset instance.
    
    Args:
        data_dir: Root directory of the dataset
        image_dir: Subdirectory containing the images
        mask_dir: Subdirectory containing the segmentation masks
        split: Either 'train' or 'val'
        transform: Albumentations transform to be applied
        max_samples: Maximum number of samples to use (for debugging)
        
    Returns:
        AutoAlbumentDataset instance
    """
    split_file = f"{split}.txt" if split in ["train", "val"] else split
    return AutoAlbumentDataset(
        root=data_dir,
        image_dir=image_dir,
        mask_dir=mask_dir,
        split_file=split_file,
        transform=transform,
        max_samples=max_samples,
        num_classes=num_classes,
    )