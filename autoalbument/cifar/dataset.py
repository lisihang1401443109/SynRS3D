import cv2
import torchvision
import torch

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    
    
class Cifar10TrainDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        
        # convert to numpy array
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        # if PIL Image
        elif isinstance(image, PIL.Image.Image):
            image = np.array(image)
        

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label