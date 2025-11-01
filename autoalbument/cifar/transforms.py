import json
from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_train_transforms(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    return A.Compose(
        [
            A.RandomCrop(height=32, width=32),
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
    return A.load(policy_json_path, data_format="json")


essential_post = [
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ToTensorV2(),
]


def get_policy_train_transforms(policy_json_path: str):
    policy = load_policy_from_json(policy_json_path)
    transforms = [
        A.RandomCrop(32, 32, always_apply=True),
        policy,
        # A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)
