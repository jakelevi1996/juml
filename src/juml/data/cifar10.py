import torch
import torch.utils.data
import torchvision
from jutility import cli
from juml.data.classification import ClassificationDataset

class Cifar10(ClassificationDataset):
    def __init__(self):
        self.split_dict = {
            "train": torchvision.datasets.CIFAR10(
                root="data",
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
            "test": torchvision.datasets.CIFAR10(
                root="data",
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
        }

    def get_split(self, split: str) -> torch.utils.data.Dataset:
        return self.split_dict[split]

    def get_input_dim(self) -> int:
        return 3*32*32

    def get_output_dim(self) -> int:
        return 10

    def format_batch(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(-3, -1)
        t = torch.nn.functional.one_hot(t, 10).float()
        return x, t
