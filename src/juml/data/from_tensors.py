import torch
import torch.utils.data

class DataSplitFromTensors(torch.utils.data.Dataset):
    def __init__(
        self,
        x_ni:   torch.Tensor,
        t_no:   torch.Tensor,
        n:      int,
    ):
        self.x_ni   = x_ni
        self.t_no   = t_no
        self.n      = n

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x_ni[i], self.t_no[i]

    def __len__(self) -> int:
        return self.n
