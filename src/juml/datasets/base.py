import torch.utils.data
from jutility import cli, util, units
from juml.datasets.split import DataSplit
from juml.datasets import loss

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
