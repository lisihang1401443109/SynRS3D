import albumentations as A
import json
import argparse

def print_structure(t, indent=0):
    prefix = "  " * indent
    name = t.__class__.__name__
    p = getattr(t, 'p', 'N/A')
    print(f"{prefix}{name} (p={p})")
    
    if hasattr(t, 'transforms'):
        for child in t.transforms:
            print_structure(child, indent + 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    
    print(f"Loading {args.policy}...")
    policy = A.load(args.policy, data_format='json')
    
    print("\nFull Policy Structure:")
    print_structure(policy)
    
    print("\nSliced Policy (2:-2):")
    if hasattr(policy, 'transforms') and len(policy.transforms) >= 4:
        sliced = policy.transforms[2:-2]
        for t in sliced:
            print_structure(t, indent=1)
    else:
        print("Cannot slice, using full policy")

if __name__ == "__main__":
    main()
