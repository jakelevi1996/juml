from juml.data.dataset import Dataset
from juml.data.classification import ClassificationDataset
from juml.data.mnist import Mnist

def get_all_datasets() -> list[type[Dataset]]:
    return [
        Mnist,
    ]
