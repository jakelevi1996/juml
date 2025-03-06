import math
import torch
from jutility import cli
from juml.models import embed, pool
from juml.models.base import Model

class LinearModel(Model):
    def __init__(
        self,
        input_shape: list[int],
        output_shape: list[int],
        embedder: embed.Embedder,
        pooler: pool.Pooler,
    ):
        self._torch_module_init()

        embedder.set_input_shape(input_shape)
        pooler.set_shapes([], output_shape)
        self.embed = embedder
        self.pool  = pooler

        self.layer = LinearLayer(
            input_dim=embedder.get_output_dim(-1),
            output_dim=pooler.get_input_dim(-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed.forward(x)
        x = self.layer.forward(x)
        x = self.pool.forward(x)
        return x

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            embed.get_cli_choice(),
            pool.get_cli_choice(),
        ]

class LinearLayer(Model):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        self._torch_module_init()
        w_scale = 1 / math.sqrt(input_dim)
        w_shape = [input_dim, output_dim]
        self.w_io = torch.nn.Parameter(torch.normal(0, w_scale, w_shape))
        self.b_o  = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x_ni: torch.Tensor) -> torch.Tensor:
        x_no = x_ni @ self.w_io + self.b_o
        return x_no

class MultiHeadLinear(Model):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        w_scale: float,
    ):
        self._torch_module_init()
        w_shape = [num_heads, input_dim, output_dim]
        b_shape = [num_heads, 1,         output_dim]
        self.w_hio = torch.nn.Parameter(torch.normal(0, w_scale, w_shape))
        self.b_h1o = torch.nn.Parameter(torch.zeros(b_shape))

    def forward(self, x_n1pi: torch.Tensor) -> torch.Tensor:
        x_nhpo = x_n1pi @ self.w_hio + self.b_h1o
        return x_nhpo
