import torch
import torch.utils.data
from juml.data.classification import ClassificationDataset
from juml.data.from_tensors import DataSplitFromTensors

class Xor(ClassificationDataset):
    def __init__(self):
        x_list = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        t_list = [
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
        ]
        x_ni = torch.tensor(x_list).float()
        t_no = torch.tensor(t_list).float()
        self.split = DataSplitFromTensors(x_ni, t_no, 4)

    def get_split(self, split: str) -> torch.utils.data.Dataset:
        return self.split

    def get_input_dim(self) -> int:
        return 2

    def get_output_dim(self) -> int:
        return 2
