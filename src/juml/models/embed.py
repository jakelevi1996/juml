import math
import torch
from jutility import cli
from juml.models.base import Model

class Embedder(Model):
    def set_input_shape(self, input_shape: list[int]):
        raise NotImplementedError()

    def get_output_shape(self) -> list[int]:
        raise NotImplementedError()

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return []

class Identity(Embedder):
    def set_input_shape(self, input_shape: list[int]):
        self._input_shape = input_shape

    def get_output_shape(self) -> list[int]:
        return self._input_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Flatten(Embedder):
    def __init__(self, num_flatten: int):
        self._num_flatten = num_flatten

    def set_input_shape(self, input_shape: list[int]):
        self._input_shape = input_shape

    def get_output_shape(self) -> list[int]:
        return [
            *self._input_shape[:-self._num_flatten],
            math.prod(self._input_shape[-self._num_flatten:]),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(-self._num_flatten, -1)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [cli.Arg("num_flatten", type=int, default=None)]

class CoordConv(Embedder):
    def set_input_shape(self, input_shape: list[int]):
        self._torch_module_init()
        c, h, w = input_shape[-3:]
        x_1w    = torch.linspace(-1, 1, w).unsqueeze(-2)
        y_h1    = torch.linspace(-1, 1, h).unsqueeze(-1)
        x_hw    = torch.tile(x_1w, [h, 1])
        y_hw    = torch.tile(y_h1, [1, w])
        xy_2hw  = torch.stack([x_hw, y_hw], dim=-3)
        self._coord_tensor = torch.nn.Parameter(xy_2hw, requires_grad=False)
        self._output_shape = input_shape[:-3] + [c + 2, h, w]

    def get_output_shape(self) -> list[int]:
        return self._output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batched_coord_tensor = torch.tile(
        #     self._coord_tensor,
        #     list(x.shape[:-3]) + [1, 1, 1],
        # )
        batched_shape = list(x.shape[:-3]) + list(self._coord_tensor.shape)
        batched_coord_tensor = self._coord_tensor.expand(batched_shape)
        return torch.concat([x, batched_coord_tensor], dim=-3)
