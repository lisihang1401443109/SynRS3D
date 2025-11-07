import cv2
import numpy as np
from torchvision.datasets import VOCSegmentation
import torch
import PIL

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCSearchDataset(VOCSegmentation):
    def __init__(self, root="/mnt/synrs3d/SynRS3D/autoalbument/pascal_voc/data", image_set="train", download=True, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
    
class PascalVOCTrainDataset(VOCSegmentation):
    def __init__(self, root="/mnt/synrs3d/SynRS3D/autoalbument/pascal_voc/data", image_set="train", download=True, transform=None):
        super().__init__(root=root, image_set=image_set, download=download, transform=transform)
        self.image_paths = self.images
        self.mask_paths = self.masks
        
    @staticmethod
    def _convert_to_segmentation_mask(mask):
        """Convert Pascal VOC RGB mask to one-hot encoded mask.
        
        Args:
            mask: numpy.ndarray, RGB mask with shape (H, W, 3)
            
        Returns:
            numpy.ndarray: One-hot encoded mask with shape (H, W, num_classes)
        """
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"Expected numpy array for mask, got {type(mask)}")
            
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
            
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        
        for label_index, label in enumerate(VOC_COLORMAP):
            # Convert both mask and label to uint8 for comparison
            label_arr = np.array(label, dtype=np.uint8).reshape(1, 1, 3)
            # Compare RGB values
            segmentation_mask[:, :, label_index] = np.all(mask == label_arr, axis=-1).astype(np.float32)
            
        return segmentation_mask
    
    def _load_image(self, path):
        """Load and validate an image."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
                
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            raise
    
    def _load_mask(self, path):
        """Load and validate a mask."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Mask file not found: {path}")
                
            mask = cv2.imread(str(path))
            if mask is None:
                raise ValueError(f"Failed to load mask: {path}")
                
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            return mask
            
        except Exception as e:
            print(f"Error loading mask {path}: {str(e)}")
            raise
        
    def __getitem__(self, index):
        try:
            # Load image and mask
            image = self._load_image(self.image_paths[index])
            mask = self._load_mask(self.mask_paths[index])
            
            # Convert mask to one-hot encoding
            mask = self._convert_to_segmentation_mask(mask)
            
            # Apply transforms if specified
            if self.transform is not None:
                # Ensure data is in the correct format for albumentations
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                    
                # Apply transformations
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
                
                # Convert to float32 if needed
                if isinstance(mask, torch.Tensor):
                    mask = mask.float()
                else:
                    mask = mask.astype(np.float32)
            
            return image, mask
            
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {str(e)}")
            # Return a zero tensor with the correct shape to avoid breaking the training loop
            # This is a fallback - you might want to implement a better strategy
            dummy_image = torch.zeros((3, 320, 320), dtype=torch.float32)
            dummy_mask = torch.zeros((len(VOC_COLORMAP), 320, 320), dtype=torch.float32)
            return dummy_image, dummy_mask
    
    def __len__(self):
        return len(self.image_paths)