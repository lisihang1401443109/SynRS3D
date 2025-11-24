import json
import argparse
import os
import albumentations as A
import cv2
import numpy as np
import tifffile

def filter_transforms(config, threshold=0.01):
    """
    Recursively filter transforms in the configuration based on probability.
    """
    if not isinstance(config, dict):
        return config

    # If it's a composition or wrapper that has 'transforms'
    if "transforms" in config:
        new_transforms = []
        for t in config["transforms"]:
            # Check probability if it exists
            p = t.get("p", 1.0)
            if p >= threshold:
                # Recursively filter children
                filtered_t = filter_transforms(t, threshold)
                new_transforms.append(filtered_t)
        config["transforms"] = new_transforms
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Visualize AutoAugment Policy")
    parser.add_argument("--policy", required=True, help="Path to the policy JSON file")
    parser.add_argument("--image", required=True, help="Path to the source image (TIFF)")
    parser.add_argument("--output", required=True, help="Output directory for transformed images")
    parser.add_argument("--threshold", type=float, default=0.01, help="Probability threshold to ignore transforms")
    parser.add_argument("--num_images", type=int, default=5, help="Number of transformed images to generate")
    
    args = parser.parse_args()

    # 1. Load Policy
    with open(args.policy, "r") as f:
        policy_data = json.load(f)

    print(f"Loaded policy from {args.policy}")

    # 2. Filter Policy
    # The policy JSON usually has a structure like {"transform": {...}} or just the transform dict
    # Based on the user's file, it starts with {"__version__": ..., "transform": {...}}
    
    if "transform" in policy_data:
        transform_config = policy_data["transform"]
    else:
        transform_config = policy_data

    filtered_config = filter_transforms(transform_config, args.threshold)
    
    print(f"Filtered transforms with threshold {args.threshold}")

    # 3. Create Pipeline
    try:
        transform = A.from_dict(filtered_config)
    except Exception as e:
        print(f"Error creating albumentations pipeline: {e}")
        # Fallback: sometimes from_dict expects the full dict including __version__ if it was serialized that way,
        # but usually it expects the transform dict. 
        # If the filtering modified it in a way A.from_dict doesn't like, we might need to be careful.
        # Let's try to reconstruct the full dict if needed, but usually passing the transform dict is correct.
        return

    # 4. Load Image
    try:
        image = tifffile.imread(args.image)
    except Exception as e:
        print(f"Error loading image with tifffile: {e}")
        # Fallback to cv2 if tifffile fails or if it's not actually a tiff
        image = cv2.imread(args.image)
        if image is None:
            print("Failed to load image.")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Loaded image with shape {image.shape}")

    # 5. Apply Transforms and Save
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save original
    cv2.imwrite(os.path.join(args.output, "original.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    for i in range(args.num_images):
        try:
            augmented = transform(image=image)["image"]
            output_path = os.path.join(args.output, f"aug_{i}.png")
            # Convert back to BGR for saving with cv2
            cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            print(f"Saved {output_path}")
        except Exception as e:
            print(f"Error applying transform {i}: {e}")

if __name__ == "__main__":
    main()
