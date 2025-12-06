import argparse
import albumentations as A
import json
import sys
import os

# Add current directory to path to import visualize_policy if needed, 
# but I'll just copy the make_deterministic logic to be standalone and safe.

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()

    try:
        policy = A.load(args.policy, data_format='json')
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    if hasattr(policy, 'transforms') and len(policy.transforms) >= 4:
        policy_list = policy.transforms[2:-2]
    else:
        policy_list = [policy]

    deterministic_list = []
    for t in policy_list:
        res = make_deterministic(t)
        if res:
            deterministic_list.append(res)
            
    print(f"Deterministic transforms for {args.policy}:")
    print(deterministic_list)

if __name__ == "__main__":
    main()
