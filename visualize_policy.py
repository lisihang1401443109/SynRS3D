import json
import argparse
import os
import albumentations as A
import cv2
import numpy as np
import tifffile

def simplify_policy(config, threshold=0.1):
    """
    Simplify the policy by deterministically selecting the maximal non-NoOp operation
    in OneOf blocks, provided it exceeds the threshold.
    """
    if not isinstance(config, dict):
        return config

    # Handle specific transforms to remove
    name = config.get("__class_fullname__", "")
    if name in ["Normalize", "ToTensorV2", "NoOp"]:
        return None

    # If it has children (Compose, Sequential, OneOf)
    if "transforms" in config:
        children = config["transforms"]
        
        if name == "OneOf":
            # Find the best non-NoOp child
            best_child = None
            max_p = -1.0
            
            for child in children:
                child_name = child.get("__class_fullname__", "")
                if child_name == "NoOp":
                    continue
                
                p = child.get("p", 0.0)
                if p > max_p:
                    max_p = p
                    best_child = child
            
            # If we found a candidate and it meets the threshold
            if best_child and max_p >= threshold:
                return simplify_policy(best_child, threshold)
            else:
                return None
        else:
            # For Sequential/Compose, simplify all children
            new_children = []
            for child in children:
                simplified_child = simplify_policy(child, threshold)
                if simplified_child is not None:
                    new_children.append(simplified_child)
            
            # If no children remain, return None (unless it's the root or something we want to keep empty?)
            # But usually an empty Compose is useless.
            if not new_children:
                return None
                
            config["transforms"] = new_children
            return config

    return config

def main():
    parser = argparse.ArgumentParser(description="Visualize AutoAugment Policy")
    parser.add_argument("--policy", required=True, help="Path to the policy JSON file")
    parser.add_argument("--image", required=True, help="Path to the source image (TIFF)")
    parser.add_argument("--output", required=True, help="Output directory for transformed images")
    parser.add_argument("--threshold", type=float, default=0.1, help="Probability threshold to select operations (default: 0.1)")
    parser.add_argument("--num_images", type=int, default=5, help="Number of transformed images to generate")
    
    args = parser.parse_args()

    # 1. Load Policy
    with open(args.policy, "r") as f:
        policy_data = json.load(f)

    print(f"Loaded policy from {args.policy}")

    # 2. Simplify Policy
    if "transform" in policy_data:
        transform_config = policy_data["transform"]
    else:
        transform_config = policy_data

    # Simplify the config
    # We work on a copy to avoid mutating the original if we needed it (though here we don't)
    import copy
    simplified_config = simplify_policy(copy.deepcopy(transform_config), args.threshold)
    
    if simplified_config is None:
        print("Warning: Policy simplified to nothing (all probabilities below threshold).")
        simplified_config = {"__class_fullname__": "Compose", "transforms": []}

    print(f"Simplified policy with threshold {args.threshold}")

    # Reconstruct the full dict for A.from_dict
    full_config = {"transform": simplified_config}
    if "__version__" in policy_data:
        full_config["__version__"] = policy_data["__version__"]

    print("Simplified Policy Configuration:")
    print(json.dumps(full_config, indent=2))

    # 3. Create Pipeline
    try:
        # print(f"DEBUG: full_config keys: {full_config.keys()}")
        transform = A.from_dict(full_config)
    except Exception as e:
        import traceback
        traceback.print_exc()
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
