import torch
from jutility import cli
from juml.datasets import loss
from juml.datasets.synthetic import Synthetic

class SinMix(Synthetic):
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int,
        output_dim: int,
        train:      int,
        test:       int,
        x_std:      float,
        t_std:      float,
    ):
        self.w1_ih = torch.normal(0, 1, [input_dim,  hidden_dim])
        self.w2_ho = torch.normal(0, 1, [hidden_dim, output_dim])
        self.b1_h  = torch.normal(0, 1, [hidden_dim])
        self.b2_o  = torch.normal(0, 1, [output_dim])

        self._init_synthetic(
            input_shape=[input_dim],
            output_shape=[output_dim],
            n_train=train,
            n_test=test,
            x_std=x_std,
            t_std=t_std,
        )

    def _forward(self, x_ni: torch.Tensor) -> torch.Tensor:
        z_nh = x_ni @ self.w1_ih + self.b1_h
        z_nh = torch.sin(z_nh)
        t_no = z_nh @ self.w2_ho + self.b2_o
        return t_no

    def _get_loss(self) -> loss.Loss:
        return loss.Mse()

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            cli.Arg("input_dim",    type=int,   default=1),
            cli.Arg("hidden_dim",   type=int,   default=1),
            cli.Arg("output_dim",   type=int,   default=1),
            cli.Arg("train",        type=int,   default=100),
            cli.Arg("test",         type=int,   default=100),
            cli.Arg("x_std",        type=float, default=0.0),
            cli.Arg("t_std",        type=float, default=0.1),
        )
