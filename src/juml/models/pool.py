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

class Average2d(Pooler):
    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        self.linear = torch.nn.Linear(input_shape[-3], output_shape[-1])

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        x_nci = x_nchw.flatten(-2, -1)
        x_nc  = x_nci.mean(dim=-1)
        x_no  = self.linear.forward(x_nc)
        return x_no

class Max2d(Pooler):
    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        self.linear = torch.nn.Linear(input_shape[-3], output_shape[-1])

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        x_nci = x_nchw.flatten(-2, -1)
        x_nc  = x_nci.max(dim=-1).values
        x_no  = self.linear.forward(x_nc)
        return x_no

class Attention2d(Pooler):
    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        self.linear = torch.nn.Linear(input_shape[-3], output_shape[-1])
        self.conv   = torch.nn.Conv2d(input_shape[-3], 1, 1)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        p_n1hw = self.conv.forward(x_nchw)
        p_n11i = p_n1hw.flatten(-2, -1).unsqueeze(-2)
        p_n11i = torch.softmax(p_n11i, dim=-1)
        x_nci1 = x_nchw.flatten(-2, -1).unsqueeze(-1)
        x_nc11 = p_n11i @ x_nci1
        x_nc   = x_nc11.squeeze(-2).squeeze(-1)
        x_no   = self.linear.forward(x_nc)
        return x_no

class Conv2d(Pooler):
    def __init__(self, *args, **kwargs):
        self._torch_module_init()

    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        self.conv = torch.nn.Conv2d(input_shape[-3], output_shape[-1], 1)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        x_nohw = self.conv.forward(x_nchw)
        return x_nohw

class SigmoidProduct2d(Pooler):
    def set_shapes(
        self,
        input_shape:  list[int],
        output_shape: list[int],
    ):
        self.conv_p = torch.nn.Conv2d(input_shape[-3], 1, 1)
        self.conv_x = torch.nn.Conv2d(input_shape[-3], output_shape[-1], 1)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        p_n1hw = self.conv_p.forward(x_nchw)
        p_n1hw = torch.sigmoid(p_n1hw)
        x_nohw = self.conv_x.forward(x_nchw)
        x_nohw = p_n1hw * x_nohw
        return x_nohw

def get_types() -> list[type[Pooler]]:
    return [
        Identity,
        Unflatten,
        Average2d,
        Max2d,
        Attention2d,
        Conv2d,
        SigmoidProduct2d,
    ]

def get_cli_choice() -> cli.ObjectChoice:
    return cli.ObjectChoice(
        "pooler",
        *[pool_type.get_cli_arg() for pool_type in get_types()],
        default="Identity",
    )
