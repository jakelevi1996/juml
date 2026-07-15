import torch
from jutility import cli

class Model(torch.nn.Module):
    def _torch_module_init(self):
        super().__init__()

    def get_num_params(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
        )

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        raise NotImplementedError()

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(
            cls,
            *cls.get_cli_options(),
            tag=cls.get_tag(),
        )

    @classmethod
    def get_tag(cls) -> (str | None):
        return None

    def __repr__(self) -> str:
        return type(self).__name__
