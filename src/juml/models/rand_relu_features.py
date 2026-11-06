import torch
from jutility import cli
from juml.models.ff import FeedForwardModel
from juml.models.layers.linear import LinearLayer

class RandReluFeatures(FeedForwardModel):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        hidden_dim: int,
        reg:        float,
    ):
        self._torch_module_init()
        self.reg = reg
        self.hidden_layer = LinearLayer(input_dim,  hidden_dim)
        self.output_layer = LinearLayer(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer.forward(x)
        x = x.relu()
        x = self.output_layer.forward(x)
        return x

    def fit(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.hidden_layer.normalise(x)
        x = self.hidden_layer.forward(x)
        x = x.relu()
        self.output_layer.lstsq(x, t, self.reg)
        x = self.output_layer.forward(x)
        return x

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("hidden_dim",   type=int,   default=100),
            cli.Arg("reg",          type=float, default=1e-3),
        ]
