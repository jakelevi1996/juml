from juml.data.dataset import Dataset
from juml.data.classification import ClassificationDataset
from juml.data.xor import Xor
from juml.data.mnist import Mnist
from juml.data.cifar10 import Cifar10

def get_all_datasets() -> list[type[Dataset]]:
    return [
        Xor,
        Mnist,
        Cifar10,
    ]
