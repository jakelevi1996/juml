import math
import torch
from jutility import cli
from juml.models import embed, pool
from juml.models.base import Model
from juml.models.linear import LinearLayer

class Mlp(Model):
    def __init__(
        self,
        input_shape: list[int],
        output_shape: list[int],
        hidden_dim: int,
        num_hidden_layers: int,
        embedder: embed.Embedder,
        pooler: pool.Pooler,
    ):
        self._torch_module_init()

        self.embed = embedder
        self.embed.set_input_shape(input_shape)

        self.pool = pooler
        self.pool.set_shapes([], output_shape)

        layer_input_dim = embedder.get_output_dim(-1)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            layer = LinearLayer(layer_input_dim, hidden_dim)
            self.hidden_layers.append(layer)
            layer_input_dim = hidden_dim

        layer_output_dim = self.pool.get_input_dim(-1)
        self.output_layer = LinearLayer(layer_input_dim, layer_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed.forward(x)

        for layer in self.hidden_layers:
            x = torch.relu(layer.forward(x))

        x = self.output_layer.forward(x)
        x = self.pool.forward(x)
        return x

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("hidden_dim",           type=int, default=100),
            cli.Arg("num_hidden_layers",    type=int, default=3),
            embed.get_cli_choice(),
            pool.get_cli_choice(),
        ]
