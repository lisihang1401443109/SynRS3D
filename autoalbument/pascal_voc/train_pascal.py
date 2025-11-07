import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import PascalVOCTrainDataset, PascalVOCSearchDataset, VOC_CLASSES
from transforms import (
    get_base_train_transforms,
    get_policy_train_transforms,
    get_test_transforms,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
# check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



@torch.no_grad()
def evaluate_miou(model, loader, device, num_classes):
    model.eval()
    ious = torch.zeros(num_classes, dtype=torch.float64, device=device)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=device)
    
    for batch_idx, batch in enumerate(loader):
        try:
            if len(batch) != 2:
                print(f"Unexpected batch format. Expected length 2, got {len(batch)}. Batch: {batch}")
                continue
                
            images, masks = batch
            if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
                print(f"Unexpected data types. Images type: {type(images)}, Masks type: {type(masks)}")
                continue
                
            images = images.to(device)
            
            # Convert mask to proper format
            if masks.dim() == 4:  # Should be (B, C, H, W) or (B, H, W, C)
                if masks.shape[1] == num_classes:  # (B, C, H, W)
                    masks = masks.argmax(dim=1)
                elif masks.shape[-1] == num_classes:  # (B, H, W, C)
                    masks = masks.permute(0, 3, 1, 2).argmax(dim=1)
                else:
                    print(f"Unexpected mask shape {tuple(masks.shape)}; expected channels to be {num_classes}")
                    continue
            elif masks.dim() != 3:  # Should be (B, H, W) after processing
                print(f"Unexpected mask dimensions: {masks.dim()}, shape: {tuple(masks.shape)}")
                continue
                
            masks = masks.long().to(device)
            
            # Forward pass
            outputs = model(images)["out"]
            preds = outputs.argmax(dim=1)
            
            # Calculate IoU for each class
            for cls in range(num_classes):
                pred_c = preds == cls
                mask_c = masks == cls
                inter = (pred_c & mask_c).sum().double()
                union = (pred_c | mask_c).sum().double()
                ious[cls] += inter
                unions[cls] += union
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate mean IoU
    valid = unions > 0
    if not valid.any():
        print("Warning: No valid classes found in validation set")
        return 0.0
        
    miou = (ious[valid] / unions[valid]).mean().item()
    return miou


from tqdm import tqdm


def build_loaders(root, batch_size, workers, train_tfms, test_tfms, download):
    print("Building data loaders...")
    print(f"Train transforms: {train_tfms}")
    print(f"Test transforms: {test_tfms}")
    
    # Create datasets
    train_set = PascalVOCTrainDataset(root=root, image_set="train", download=download, transform=train_tfms)
    val_set = PascalVOCTrainDataset(root=root, image_set="val", download=download, transform=test_tfms)
    
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    
    # Create data loaders with error handling
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, workers),  # Limit workers to avoid issues
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch
        persistent_workers=workers > 0  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, workers),  # Use fewer workers for validation
        pin_memory=True,
        drop_last=False,  # Don't drop last batch in validation
        persistent_workers=workers > 0
    )
    
    return train_loader, val_loader


def build_model(num_classes, pretrained):
    if pretrained:
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    else:
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--policy_json", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--only_policy", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=str(Path("./outputs/pascal").resolve()))
    parser.add_argument("--log_dir", type=str, default=str(Path("./runs/pascal").resolve()))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    num_classes = len(VOC_CLASSES)
    test_tfms = get_test_transforms(size=args.size)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    results = {}

    if not args.only_policy:
        base_tfms = get_base_train_transforms(size=args.size)
        print(base_tfms)
        base_train_loader, base_val_loader = build_loaders(
            args.data_root, args.batch_size, args.workers, base_tfms, test_tfms, args.download
        )
        print(base_train_loader)
        base_model = build_model(num_classes=num_classes, pretrained=args.pretrained)
        base_best = train_one(
            base_model,
            base_train_loader,
            base_val_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            num_classes,
            writer,
            os.path.join(args.output_dir, "pascal_baseline_best.pth"),
        )
        results["baseline_best_miou"] = base_best

    if args.policy_json and os.path.isfile(args.policy_json):
        policy_tfms = get_policy_train_transforms(args.policy_json, size=args.size)
        print(policy_tfms)
        pol_train_loader, pol_val_loader = build_loaders(
            args.data_root, args.batch_size, args.workers, policy_tfms, test_tfms, args.download
        )
        print(pol_train_loader)
        pol_model = build_model(num_classes=num_classes, pretrained=args.pretrained)
        pol_best = train_one(
            pol_model,
            pol_train_loader,
            pol_val_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            num_classes,
            writer,
            os.path.join(args.output_dir, "pascal_policy_best.pth"),
        )
        results["policy_best_miou"] = pol_best

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f)
    print(json.dumps(results))
    writer.close()


if __name__ == "__main__":
    main()
