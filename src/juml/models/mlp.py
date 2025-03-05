import math
import torch
from jutility import cli
from juml.models.base import Model
from juml.models.linear import LinearLayer

class Mlp(Model):
    def __init__(
        self,
        input_shape: list[int],
        output_shape: list[int],
        hidden_dim: int,
        num_hidden_layers: int,
        num_flattened_input_dims: int,
    ):
        self._torch_module_init()
        self.num_flat = num_flattened_input_dims
        self.hidden_layers = torch.nn.ModuleList()

        layer_input_dim = math.prod(input_shape[-self.num_flat:])
        for _ in range(num_hidden_layers):
            layer = LinearLayer(layer_input_dim, hidden_dim)
            self.hidden_layers.append(layer)
            layer_input_dim = hidden_dim

        self.output_layer = LinearLayer(layer_input_dim, output_shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(-self.num_flat, -1)

        for layer in self.hidden_layers:
            x = torch.relu(layer.forward(x))

        return self.output_layer.forward(x)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("hidden_dim",           type=int, default=100),
            cli.Arg("num_hidden_layers",    type=int, default=3),
            cli.NoTagArg("num_flattened_input_dims", type=int, default=1),
        ]
