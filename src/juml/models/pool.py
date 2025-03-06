import math
import torch
from jutility import cli
from juml.models.base import Model

class Pooler(Model):
    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        raise NotImplementedError()

    def get_input_shape(self) -> list[int]:
        raise NotImplementedError()

    def get_input_dim(self, dim: int) -> int:
        input_shape = self.get_input_shape()
        return input_shape[dim]

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return []

class Identity(Pooler):
    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        self._output_shape = output_shape

    def get_input_shape(self) -> list[int]:
        return self._output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Unflatten(Pooler):
    def __init__(self, num_unflatten: int):
        self._torch_module_init()
        self._num_unflatten = num_unflatten

    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        self._unflatten_shape = output_shape[-self._num_unflatten:]
        self._input_shape = [
            *output_shape[:-self._num_unflatten],
            math.prod(self._unflatten_shape),
        ]

    def get_input_shape(self) -> list[int]:
        return self._input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unflatten(-1, self._unflatten_shape)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [cli.Arg("num_unflatten", type=int, default=None)]

def get_types() -> list[type[Pooler]]:
    return [
        Identity,
        Unflatten,
    ]

def get_cli_choice() -> cli.ObjectChoice:
    return cli.ObjectChoice(
        "pooler",
        *[pool_type.get_cli_arg() for pool_type in get_types()],
        default="Identity",
    )
