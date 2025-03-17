import torch
from jutility import cli
from juml.datasets.split import DataSplit
from juml.datasets.synthetic import Synthetic

class RandomImage(Synthetic):
    def __init__(
        self,
        input_shape:    list[int],
        num_classes:    int,
        train:          int,
        test:           int,
    ):
        self._num_classes = num_classes
        self._init_synthetic(
            input_shape=input_shape,
            output_shape=[num_classes],
            n_train=train,
            n_test=test,
            x_std=0,
            t_std=0,
        )

    def _make_split(self, n: int) -> DataSplit:
        return DataSplit(
            x=torch.rand([n, *self._input_shape]),
            t=torch.randint(0, self._num_classes, [n]),
            n=n,
        )

    def get_default_loss(self) -> str | None:
        return "CrossEntropy"

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            cli.Arg("input_shape",  type=int, nargs="+", default=[3, 32, 32]),
            cli.Arg("num_classes",  type=int, default=10),
            cli.Arg("train",        type=int, default=200),
            cli.Arg("test",         type=int, default=200),
        )
