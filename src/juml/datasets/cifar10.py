import torch.utils.data
import torchvision
from juml.datasets import loss
from juml.datasets.base import DatasetFromDict, DATA_REL_DIR

class Cifar10(DatasetFromDict):
    def _get_split_dict(self) -> dict[str, torch.utils.data.Dataset]:
        return {
            "train": torchvision.datasets.CIFAR10(
                root=DATA_REL_DIR,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
            "test": torchvision.datasets.CIFAR10(
                root=DATA_REL_DIR,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
        }

    def get_input_shape(self) -> list[int]:
        return [3, 32, 32]

    def get_output_shape(self) -> list[int]:
        return [10]

    def _get_loss(self) -> loss.Loss:
        return loss.CrossEntropy()
