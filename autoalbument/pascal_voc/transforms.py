import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_base_train_transforms(size=320, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return A.Compose(
        [
            A.RandomCrop(size, size, always_apply=True),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_test_transforms(size=320, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def load_policy_from_json(policy_json_path: str) -> A.BasicTransform:
    return A.load(policy_json_path, data_format="json")
    
    
essential_post = [
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]


def get_policy_train_transforms(policy_json_path: str, size=320):
    policy = load_policy_from_json(policy_json_path)
    transforms = [
        A.RandomCrop(size, size, always_apply=True),
        policy,
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)
