import torch.utils.data
from juml.datasets.base import Dataset

class DatasetFromDict(Dataset):
    def __init__(self):
        self._init_split_dict()

    def _init_split_dict(self):
        self._split_dict = self._get_split_dict()

    def _get_split_dict(self) -> dict[str, torch.utils.data.Dataset]:
        raise NotImplementedError()

    def get_data_split(self, split: str) -> torch.utils.data.Dataset:
        return self._split_dict[split]
