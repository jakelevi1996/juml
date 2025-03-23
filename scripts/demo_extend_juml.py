import torch
from jutility import cli
import juml

class PolynomialRegression1d(juml.base.Model):
    def __init__(
        self,
        n:              int,
        input_shape:    list[int],
        output_shape:   list[int],
    ):
        self._torch_module_init()
        self.p_i    = torch.arange(n)
        self.w_i1   = torch.nn.Parameter(torch.zeros([n, 1]))

    def forward(self, x_n1: torch.Tensor) -> torch.Tensor:
        x_ni = (x_n1 ** self.p_i)
        x_n1 = x_ni @ self.w_i1
        return x_n1

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [cli.Arg("n", type=int, default=5)]

class Step1d(juml.datasets.Synthetic):
    def __init__(self):
        self._init_synthetic(
            input_shape=[1],
            output_shape=[1],
            n_train=200,
            n_test=200,
            x_std=0.1,
            t_std=0.02,
        )

    def _compute_target(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, 1.0, 0.0)

    def get_default_loss(self) -> type[juml.base.Loss] | None:
        return juml.loss.Mse

class DemoExtendFramework(juml.base.Framework):
    @classmethod
    def get_models(cls) -> list[type[juml.base.Model]]:
        return [
            *juml.models.get_all(),
            PolynomialRegression1d,
        ]

    @classmethod
    def get_datasets(cls) -> list[type[juml.base.Dataset]]:
        return [
            *juml.datasets.get_all(),
            Step1d,
        ]

if __name__ == "__main__":
    DemoExtendFramework.run()
