import json

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_train_transforms(size=320, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=[0, 0, 0]),
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
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=[0, 0, 0]),
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
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]


def get_policy_train_transforms(policy_json_path: str, size=320):
    policy_t = load_policy_from_json(policy_json_path)
    resize_pad = [A.LongestMaxSize(max_size=size), A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=[0, 0, 0])]
    if isinstance(policy_t, A.Compose):
        transforms = resize_pad + list(policy_t.transforms) + essential_post
        return A.Compose(transforms)
    else:
        return A.Compose(resize_pad + [policy_t] + essential_post)
