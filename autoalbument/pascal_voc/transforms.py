import json

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_train_transforms(size=320, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, border_value=[0, 0, 0]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_test_transforms(size=320, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, border_value=[0, 0, 0]),
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
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]


def get_policy_train_transforms(policy_json_path: str, size=320):
    policy = A.load(policy_json_path, data_format="json")
    transforms = [
        A.RandomCrop(size, size, always_apply=True),
        policy,
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)
