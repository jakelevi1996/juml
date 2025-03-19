import torch
from jutility import cli
from juml.datasets.split import DataSplit
from juml.datasets.synthetic import Synthetic

class RandomImage(Synthetic):
    def __init__(
        self,
        input_shape:    list[int],
        output_shape:   list[int],
        train:          int,
        test:           int,
        output_float:   bool
    ):
        self._output_float = output_float
        self._init_synthetic(
            input_shape=input_shape,
            output_shape=output_shape,
            n_train=train,
            n_test=test,
            x_std=0,
            t_std=0,
        )

    def _make_split(self, n: int) -> DataSplit:
        return DataSplit(
            x=torch.rand([n, *self._input_shape]),
            t=(
                torch.rand([n, *self._output_shape])
                if self._output_float else
                torch.randint(
                    low=0,
                    high=self._output_shape[-1],
                    size=[n, *self._output_shape[:-1]],
                )
            ),
            n=n,
        )

    def get_default_loss(self) -> str | None:
        return "Mse" if self._output_float else "CrossEntropy"

    def get_loss_weights(self) -> torch.Tensor:
        split_dict  = self._get_split_dict()
        train_split = split_dict["train"]
        return train_split.t.flatten(0, -2).mean(-2)

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            cli.Arg("input_shape",  type=int, nargs="+", default=[3, 32, 32]),
            cli.Arg("output_shape", type=int, nargs="+", default=[10]),
            cli.Arg("train",        type=int, default=200),
            cli.Arg("test",         type=int, default=200),
            cli.Arg("output_float", action="store_true", tag="f"),
        )
