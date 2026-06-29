import torch
import torch.utils.data
import torchvision
from jutility import cli
from juml.data.classification import ClassificationDataset

class Mnist(ClassificationDataset):
    def __init__(self):
        self.split_dict = {
            "train": torchvision.datasets.MNIST(
                root="data",
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
            "test": torchvision.datasets.MNIST(
                root="data",
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
        }

    def get_split(self, split: str) -> torch.utils.data.Dataset:
        return self.split_dict[split]

    def get_input_dim(self) -> int:
        return 28*28

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
