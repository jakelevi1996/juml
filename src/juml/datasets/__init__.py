DATA_REL_DIR = "./data"

from juml.datasets.base import Dataset
from juml.datasets.split import DataSplit
from juml.datasets.fromdict import DatasetFromDict
from juml.datasets.synthetic import Synthetic
from juml.datasets.linear import LinearDataset
from juml.datasets.sinmix import SinMix
from juml.datasets.randimg import RandomImage
from juml.datasets.mnist import Mnist
from juml.datasets.cifar10 import Cifar10

def get_all() -> list[type[Dataset]]:
    return [
        LinearDataset,
        SinMix,
        RandomImage,
        Mnist,
        Cifar10,
    ]
