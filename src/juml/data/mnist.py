import torch
import torch.utils.data
import torchvision
from jutility import cli
from juml.data.dataset import Dataset

class Mnist(Dataset):
    def __init__(self, flat: bool):
        self.flat = flat
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

    def get_dimensions(self) -> list[list[int]]:
        input_shape = [28*28] if self.flat else [1, 28, 28]
        return [input_shape, [10]]

    def format_batch(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flat:
            x = x.flatten(-3, -1)

        t = torch.nn.functional.one_hot(t, 10).float()
        return x, t

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("flat", action="store_true"),
        ]
