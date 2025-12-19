import json
import argparse
import albumentations as A
import cv2
import sys
import os
import tifffile
import numpy as np

# Helper to serialize Albumentations transforms to dict/JSON
def transform_to_dict(transform):
    """
    Manually convert a transform to a dictionary representation safe for JSON.
    We avoid A.to_dict() sometimes because compatibility issues or verbosity.
    But A.to_dict() is standard. Let's try to construct a clean dict.
    """
    return A.to_dict(transform)

def get_class_name(t):
    return t.__class__.__name__

def simplify_transform(t):
    """
    Recursively select the highest probability branch.
    Returns a single transform or a list of transforms (if Sequential).
    """
    if isinstance(t, A.Compose):
        # Flatten compose
        new_transforms = []
        for x in t.transforms:
            res = simplify_transform(x)
            if res:
                if isinstance(res, list):
                    new_transforms.extend(res)
                else:
                    new_transforms.append(res)
        return A.Compose(new_transforms)

    if isinstance(t, A.OneOf):
        # Pick highest probability child
        # Note: In AutoAlbument, the OneOf children have their own probabilities.
        # We assume we pick one.
        if not t.transforms:
            return None
        
        # Sort by p descending
        sorted_children = sorted(t.transforms, key=lambda x: x.p, reverse=True)
        best = sorted_children[0]
        
        # Recurse on the best child
        return simplify_transform(best)
    
    if isinstance(t, A.Sequential):
        new_transforms = []
        for x in t.transforms:
            res = simplify_transform(x)
            if res:
                if isinstance(res, list):
                    new_transforms.extend(res)
                else:
                    new_transforms.append(res)
        return new_transforms

    # Base case: Any other transform (like RandomBrightnessContrast, or NoOp)
    # We treat it as deterministic (p=1) for the visualization "path"
    # But if it's NoOp, we might want to skip it?
    if get_class_name(t) == 'NoOp':
        return None
    
    # Clone and set p=1
    # We can't easily clone without deepcopy, so we'll just return it and assume usage implies determinism
    # t.p = 1.0 # mutating might be dangerous if reused, but okay for this script
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    # Load Full Policy
    try:
        policy = A.load(args.policy, data_format='json')
    except Exception as e:
        print(f"Error loading {args.policy}: {e}")
        # Try finding the 'transform' key if it's a wrapped dict
        with open(args.policy) as f:
            d = json.load(f)
        if 'transform' in d:
             # This is a bit hacky, but A.load expects the dict.
             # If A.load failed, maybe data_format='json' handled the file read but the content was issue.
             # Let's re-raise for now.
             raise e

    # Simplify
    # AutoAlbument policy is usually Compose([LongestMaxSize, PadIfNeeded, OneOf(...), Normalize, ToTensorV2])
    # The user wants "Highest probability operation".
    
    # Let's apply our recursive simplifier
    simplified = simplify_transform(policy)
    
    # Ensure result is a list or Compose
    final_transforms = []
    if isinstance(simplified, A.Compose):
        final_transforms = simplified.transforms
    elif isinstance(simplified, list):
        final_transforms = simplified
    else:
        if simplified:
            final_transforms = [simplified]
    
    # Filter out Normalize and ToTensorV2 for visualization/readability if desired
    # The user requested "simplification", usually implies readability.
    filtered_transforms = []
    for t in final_transforms:
        name = get_class_name(t)
        if name in ['Normalize', 'ToTensorV2']:
            continue
        filtered_transforms.append(t)
    
    final_pipeline = A.Compose(filtered_transforms)
    
    # Save the Simplified JSON
    simplified_dict = A.to_dict(final_pipeline)
    with open(args.output_json, 'w') as f:
        json.dump(simplified_dict, f, indent=4)
        
    print(f"Saved simplified policy to {args.output_json}")
    
    # Visualization
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    try:
        image = tifffile.imread(args.image)
    except:
        image = cv2.imread(args.image)
        if image is not None:
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        print("Failed to load image")
        sys.exit(1)

    # Save original
    cv2.imwrite(os.path.join(args.output_dir, "original.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Apply transform 5 times (though it should be deterministic now?)
    # If the transforms themselves have random parameters (like ranges), it might still vary.
    # But the *sequence* of operations is fixed.
    
    for i in range(5):
        aug = final_pipeline(image=image)["image"]
        cv2.imwrite(os.path.join(args.output_dir, f"aug_{i}.png"), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
        
    print("Visualization saved.")

if __name__ == "__main__":
    main()
