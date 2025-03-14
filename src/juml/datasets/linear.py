import torch
from jutility import cli
from juml.datasets import loss
from juml.datasets.synthetic import Synthetic

class LinearDataset(Synthetic):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        train:      int,
        test:       int,
        x_std:      float,
        t_std:      float,
    ):
        self.w_io = torch.normal(0, 1, [input_dim, output_dim])
        self.b_o  = torch.normal(0, 1, [output_dim])

        self._init_synthetic(
            input_shape=[input_dim],
            output_shape=[output_dim],
            n_train=train,
            n_test=test,
            x_std=x_std,
            t_std=t_std,
        )

    def _compute_target(self, x_ni: torch.Tensor) -> torch.Tensor:
        t_no = x_ni @ self.w_io + self.b_o
        return t_no

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
