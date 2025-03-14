import torch
from jutility import cli
from juml.datasets import loss
from juml.datasets.split import DataSplit
from juml.datasets.fromdict import DatasetFromDict

class LinearDataset(DatasetFromDict):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        train:      int,
        test:       int,
        x_std:      float,
        t_std:      float,
    ):
        self._init_loss()

        self._input_dim  = input_dim
        self._output_dim = output_dim

        self.w_io = torch.normal(0, 1, [input_dim, output_dim])
        self.b_o  = torch.normal(0, 1, [output_dim])

        self._split_dict = {
            "train": self._make_split(train, x_std, t_std),
            "test":  self._make_split(test,  x_std, t_std),
        }

    def _make_split(
        self,
        n:      int,
        x_std:  float,
        t_std:  float,
    ):
        x_ni = torch.normal(0, 1, [n, self._input_dim])
        t_no = x_ni @ self.w_io + self.b_o
        return DataSplit(
            x=x_ni + torch.normal(0, x_std, x_ni.shape),
            t=t_no + torch.normal(0, t_std, t_no.shape),
            n=n,
        )

    def get_input_shape(self) -> list[int]:
        return [self._input_dim]

    def get_output_shape(self) -> list[int]:
        return [self._output_dim]

    def _get_loss(self) -> loss.Loss:
        return loss.Mse()

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            cli.Arg("input_dim",    type=int,   default=10),
            cli.Arg("output_dim",   type=int,   default=10),
            cli.Arg("train",        type=int,   default=1000),
            cli.Arg("test",         type=int,   default=1000),
            cli.Arg("x_std",        type=float, default=0.0),
            cli.Arg("t_std",        type=float, default=0.0),
        )
