import torch
from jutility import cli
import juml
from juml.models.classification import FeedForwardModel
from juml.models.layers.linear import LinearLayer

class ReluMlp(FeedForwardModel):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        hidden_dim: int,
        depth:      int,
    ):
        self._torch_module_init()
        self.hidden_layers = torch.nn.ModuleList()
        layer_input_dim = input_dim

        for _ in range(depth - 1):
            layer = LinearLayer(layer_input_dim, hidden_dim)
            self.hidden_layers.append(layer)
            layer_input_dim = hidden_dim

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
            cli.Arg("hidden_dim",   type=int, default=100),
            cli.Arg("depth",        type=int, default=2),
        ]
