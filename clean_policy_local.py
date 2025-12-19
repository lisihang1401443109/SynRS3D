import json
import argparse

def clean_policy(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)
    
    # We expect data['transform']['transforms'] to be a list
    transforms = data['transform']['transforms']
    new_transforms = []
    
    for t in transforms:
        name = t.get('__class_fullname__')
        # Remove Normalize and ToTensorV2 which cause version issues or are unwanted
        if name in ['Normalize', 'ToTensorV2']:
            continue
        new_transforms.append(t)
        
    data['transform']['transforms'] = new_transforms
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    clean_policy(args.input, args.output)
