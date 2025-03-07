import torch
from jutility import util, cli, units
from juml.models.base  import Model
from juml.models import embed, pool

class Sequential(Model):
    def _init_sequential(
        self,
        embedder: embed.Embedder,
        pooler: pool.Pooler,
    ):
        self._torch_module_init()
        self.embed  = embedder
        self.layers = torch.nn.ModuleList()
        self.pool   = pooler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed.forward(x)

        for layer in self.layers:
            x = layer.forward(x)

        x = self.pool.forward(x)
        return x

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            *cls.get_cli_options(),
            embed.get_cli_choice(),
            pool.get_cli_choice(),
        )
