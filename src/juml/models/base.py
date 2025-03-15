import torch
from jutility import util, cli, units

class Model(torch.nn.Module):
    def _torch_module_init(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def init_batch(self, x: torch.Tensor, t: torch.Tensor):
        return

    def num_params(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            *cls.get_cli_options(),
        )

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        raise NotImplementedError()

    def __repr__(self):
        return util.format_type(
            type(self),
            num_params=units.metric.format(self.num_params()),
            item_fmt="%s=%s",
        )
