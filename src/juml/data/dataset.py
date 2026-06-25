import torch
import torch.utils.data
from jutility import cli

class Dataset:
    def get_split(self, split: str) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def get_dimensions(self) -> list[list[int]]:
        raise NotImplementedError()

    def format_batch(
        self,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return tensors

    def get_data_loader(
        self,
        split:      str,
        batch_size: int,
        shuffle:    bool=True,
    ) -> list[tuple[torch.Tensor, ...]]:
        return torch.utils.data.DataLoader(
            dataset=self.get_split(split),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_full_batch(self, split: str) -> tuple[torch.Tensor, ...]:
        data_loader = self.get_data_loader(
            split=split,
            batch_size=len(self.get_split(split)),
            shuffle=False,
        )
        tensors = next(iter(data_loader))
        tensors = self.format_batch(*tensors)
        return tensors

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            *cls.get_cli_options(),
            tag=cls.get_tag(),
        )

    @classmethod
    def get_tag(cls) -> (str | None):
        return None

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return []
