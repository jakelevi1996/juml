import torch.utils.data
import torchvision
from juml.datasets import loss
from juml.datasets.base import (
    Dataset,
    SPLIT_STR_TO_TRAIN_BOOL,
    DATA_REL_DIR,
)

class Cifar10(Dataset):
    def get_data_split(self, split: str) -> torch.utils.data.Dataset:
        return torchvision.datasets.CIFAR10(
            root=DATA_REL_DIR,
            train=SPLIT_STR_TO_TRAIN_BOOL[split],
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

    def get_input_shape(self) -> list[int]:
        return [3, 32, 32]

    def get_output_shape(self) -> list[int]:
        return [10]

    def _get_loss(self) -> loss.Loss:
        return loss.CrossEntropy()
