import json
import argparse
import os
import albumentations as A
import cv2
import numpy as np
import tifffile

def make_deterministic(transform):
    """
    Recursively select most probable transforms (ignoring NoOp) and set p=1.
    Returns the selected transform or None if it should be removed (e.g. NoOp).
    """
    name = transform.__class__.__name__
    
    if name == 'NoOp':
        return None
        
    if isinstance(transform, A.OneOf):
        # Filter NoOp
        candidates = [t for t in transform.transforms if t.__class__.__name__ != 'NoOp']
        if not candidates:
            return None
        # Pick max p
        best_child = max(candidates, key=lambda t: t.p)
        return make_deterministic(best_child)
        
    if hasattr(transform, "transforms"):
        # Sequential, Compose, etc.
        new_transforms = []
        for t in transform.transforms:
            res = make_deterministic(t)
            if res:
                new_transforms.append(res)
        transform.transforms = new_transforms
        transform.p = 1.0
        return transform
        
    # Atomic
    transform.p = 1.0
    return transform

def main():
    parser = argparse.ArgumentParser(description="Visualize AutoAugment Policy")
    parser.add_argument("--policy", required=True, help="Path to the policy JSON file")
    parser.add_argument("--image", required=True, help="Path to the source image (TIFF)")
    parser.add_argument("--output", required=True, help="Output directory for transformed images")
    parser.add_argument("--threshold", type=float, default=0.1, help="Probability threshold to select operations (default: 0.1)")
    parser.add_argument("--num_images", type=int, default=5, help="Number of transformed images to generate")
    
    args = parser.parse_args()

    # 2. Load Policy using A.load as suggested
    try:
        policy = A.load(args.policy, data_format='json')
        print(f"Loaded policy from {args.policy}")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # # 3. Create Pipeline
    # # User suggestion: policy_list = policy.transforms[2:-2] # remove the non-essence ones
    # try:
    #     if hasattr(policy, 'transforms'):
    #         # Check if we have enough transforms to slice
    #         if len(policy.transforms) >= 4:
    #             policy_list = policy.transforms[2:-2]
    #             print(f"Sliced policy transforms, kept {len(policy_list)} transforms")
    #         else:
    #             policy_list = policy.transforms
    #             print(f"Policy has fewer than 4 transforms, using all {len(policy_list)}")
    #     else:
    #         # Fallback if it's not a Compose or doesn't have transforms list in the expected way
    #         # But A.load usually returns a Compose
    #         policy_list = [policy]
    #         print("Policy is not a Compose or has no transforms list, using as is")
    policy = A.load(args.policy, data_format='json')
    # print the policy
    print(policy)
    policy_list = policy.transforms[2:-2] # remove the non-essence ones     

    # Apply deterministic selection
    deterministic_list = []
    for t in policy_list:
        res = make_deterministic(t)
        if res:
            deterministic_list.append(res)
            
    print(f"Deterministic transforms: {deterministic_list}")

    # Create a list of transforms for visualization
    # We include RandomCrop as in the snippet, but we might want to make crop size configurable or match image size
    # The snippet uses crop_size=392.
    # We will omit Normalize and ToTensorV2 for visualization purposes to keep images viewable
        
    crop_size = 392
    transforms = [
        A.RandomCrop(crop_size, crop_size),
        *deterministic_list
    ]
        
    transform = A.Compose(transforms)
        
    # except Exception as e:
    #     print(f"Error creating pipeline: {e}")
    #     return

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
