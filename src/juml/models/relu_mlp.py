import torch
from jutility import cli
import juml
from juml.models.model import Model
from juml.models.layers.linear import LinearLayer

class ReluMlp(Model):
    def __init__(
        self,
        input_shape:    list[int],
        output_shape:   list[int],
        depth:          int,
        hidden_dim:     int,
    ):
        self._torch_module_init()
        self.hidden_layers = torch.nn.ModuleList()
        [layer_input_dim] = input_shape

        for _ in range(depth - 1):
            layer = LinearLayer(layer_input_dim, hidden_dim)
            self.hidden_layers.append(layer)
            layer_input_dim = hidden_dim

        [output_dim] = output_shape
        self.output_layer = LinearLayer(layer_input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer.forward(x)
            x = torch.relu(x)

        y = self.output_layer.forward(x)
        return y

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("depth",        type=int, default=2),
            cli.Arg("hidden_dim",   type=int, default=100),
        ]
