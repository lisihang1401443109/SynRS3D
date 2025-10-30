import json
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_train_transforms(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    return A.Compose(
        [
            A.PadIfNeeded(min_height=36, min_width=36, border_mode=0, value=[0, 0, 0]),
            A.RandomCrop(height=32, width=32),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_test_transforms(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def load_policy_from_json(policy_json_path: str) -> A.BasicTransform:
    with open(policy_json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "transform" in data and isinstance(data["transform"], dict):
        transform_dict = data["transform"]
    else:
        transform_dict = data

    try:
        from albumentations.core.serialization import from_dict  # type: ignore

        transform = from_dict(transform_dict)
    except Exception:
        if hasattr(A, "from_dict"):
            transform = A.from_dict(transform_dict)  # type: ignore
        else:
            raise RuntimeError(
                "Failed to deserialize augmentation policy JSON into an Albumentations transform."
            )

    return transform


essential_post = [
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ToTensorV2(),
]


def get_policy_train_transforms(policy_json_path: str):
    policy_t = load_policy_from_json(policy_json_path)

    if isinstance(policy_t, A.Compose):
        transforms = list(policy_t.transforms) + essential_post
        return A.Compose(transforms)
    else:
        return A.Compose([policy_t] + essential_post)
