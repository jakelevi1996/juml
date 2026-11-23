import torch.utils.data
import torchvision
from jutility import cli
from juml.data.mnist import Mnist
from juml.data.from_tensors import DataSplitFromTensors

class MnistSubset(Mnist):
    def __init__(
        self,
        train_subset_size: int,
        test_subset_size:  int,
    ):
        self.split_dict = {
            "train": self.get_mnist_subset(
                train=True,
                subset_size=train_subset_size,
            ),
            "test": self.get_mnist_subset(
                train=False,
                subset_size=test_subset_size,
            ),
        }

    @classmethod
    def get_mnist_subset(
        cls,
        train:          bool,
        subset_size:    int,
    ) -> DataSplitFromTensors:
        full_split = torchvision.datasets.MNIST(
            root=cls.get_data_dir(),
            train=train,
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=full_split,
            batch_size=subset_size,
            shuffle=True,
        )
        x, t = next(iter(data_loader))
        data_subset = DataSplitFromTensors(x, t, subset_size)
        return data_subset

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("train_subset_size", type=int, default=100),
            cli.Arg("test_subset_size",  type=int, default=100),
        ]

    @classmethod
    def get_tag(cls) -> (str | None):
        return "MS"
