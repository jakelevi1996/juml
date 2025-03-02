import torch
import torch.utils.data
from jutility import cli, util, units
from juml.datasets import loss

DATA_REL_DIR = "./data"

SPLIT_STR_TO_TRAIN_BOOL = {
    "train": True,
    "test":  False,
}

class Dataset:
    def __init__(self):
        self._init_loss()

    def _init_loss(self):
        self.loss = self._get_loss()

    def get_data_loader(
        self,
        split: str,
        batch_size: int,
        shuffle=True,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        data_split = self.get_data_split(split)
        return torch.utils.data.DataLoader(
            dataset=data_split,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def get_subset_loader(
        self,
        split: str,
        subset_size: int,
        batch_size: int,
        shuffle=True,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        full_data_loader = self.get_data_loader(
            split=split,
            batch_size=subset_size,
            shuffle=True,
            **kwargs,
        )
        x, t = next(iter(full_data_loader))
        data_split = DataSplit(x, t, subset_size)
        return torch.utils.data.DataLoader(
            dataset=data_split,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def get_data_split(self, split: str) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def get_input_shape(self) -> list[int]:
        raise NotImplementedError()

    def get_output_shape(self) -> list[int]:
        raise NotImplementedError()

    def _get_loss(self) -> loss.Loss:
        raise NotImplementedError()

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(cls)

    def __repr__(self):
        return util.format_type(
            type(self),
            n_train=units.metric.format(len(self.get_data_split("train"))),
            n_test =units.metric.format(len(self.get_data_split("test" ))),
            item_fmt="%s=%s",
            key_order=["n_train", "n_test"],
        )

class DataSplit(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, t: torch.Tensor, n: int):
        self.x = x
        self.t = t
        self.n = n

    def __getitem__(self, index):
        return self.x[index], self.t[index]

    def __len__(self):
        return self.n

    def __repr__(self):
        return util.format_type(
            type(self),
            x_shape=list(self.x.shape),
            t_shape=list(self.t.shape),
            n=self.n,
        )
