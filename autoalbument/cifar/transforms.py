import json
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_train_transforms(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    return A.Compose(
        [
            A.PadIfNeeded(min_height=36, min_width=36, border_mode=0, border_value=[0, 0, 0]),
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
    try:
        return A.load(policy_json_path, data_format="json")
    except Exception:
        pass

    with open(policy_json_path, "r") as f:
        data = json.load(f)

    try:
        from albumentations.core.serialization import from_dict  # type: ignore
        payload = data if (isinstance(data, dict) and "transform" in data) else {"transform": data}
        return from_dict(payload)
    except Exception:
        if hasattr(A, "from_dict"):
            return A.from_dict(data)  # type: ignore
        raise RuntimeError("Failed to deserialize augmentation policy JSON into an Albumentations transform.")


essential_post = [
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ToTensorV2(),
]


def get_policy_train_transforms(policy_json_path: str):
    policy = A.load(policy_json_path, data_format="json")
    transforms = [
        A.RandomCrop(32, 32, always_apply=True),
        policy,
        A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)
