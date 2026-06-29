import torch
import torch.utils.data
from jutility import cli
from juml.data.dataset import Dataset

class ClassificationDataset(Dataset):
    def get_input_dim(self) -> int:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        raise NotImplementedError()

    def format_batch(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, t
