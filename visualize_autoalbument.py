import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from glob import glob
from tqdm import tqdm

def load_image(image_path):
    """Load an image from file."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_autoalbument_transforms(policy_path, crop_size=392):
    """Load and configure AutoAlbument policy."""
    policy = A.load(policy_path, data_format='json')
    print("Loaded policy:", policy)
    
    transforms = A.Compose([
        A.RandomCrop(crop_size, crop_size, p=1.0),
        policy
    ])
    return transforms

def visualize_transformations(image_path, transform, num_samples=5):
    """Visualize original and transformed images."""
    # Load the original image
    original = load_image(image_path)
    
    # Create a figure to display the results
    plt.figure(figsize=(20, 10))
    
    # Display original image
    plt.subplot(1, num_samples + 1, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')
    
    # Generate and display transformed images
    for i in range(num_samples):
        transformed = transform(image=original)['image']
        plt.subplot(1, num_samples + 1, i + 2)
        plt.imshow(transformed)
        plt.title(f"Transformed {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize AutoAlbument transformations')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images to transform')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to AutoAlbument policy JSON file')
    parser.add_argument('--crop_size', type=int, default=392,
                        help='Size for random crop (default: 392)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of transformed samples to generate per image (default: 5)')
    
    args = parser.parse_args()
    
    # Get list of images
    image_paths = glob(os.path.join(args.image_dir, '*.jpg')) + \
                 glob(os.path.join(args.image_dir, '*.png')) + \
                 glob(os.path.join(args.image_dir, '*.jpeg'))
    
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")
    
    # Load the transformation pipeline
    transform = get_autoalbument_transforms(args.policy_path, args.crop_size)
    
    # Visualize transformations for each image
    for img_path in image_paths[:5]:  # Limit to first 5 images
        print(f"\nVisualizing transformations for: {os.path.basename(img_path)}")
        try:
            visualize_transformations(img_path, transform, args.num_samples)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

if __name__ == '__main__':
    main()
