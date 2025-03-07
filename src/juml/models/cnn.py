import math
import torch
from jutility import cli
from juml.models.base import Model
from juml.models.sequential import Sequential
from juml.models import embed, pool

class Cnn(Sequential):
    def __init__(
        self,
        input_shape: list[int],
        output_shape: list[int],
        kernel_size: int,
        channel_dim: int,
        blocks: list[int],
        stride: int,
        embedder: embed.Embedder,
        pooler: pool.Pooler,
    ):
        self._init_sequential(embedder, pooler)
        self.embed.set_input_shape(input_shape)
        self.pool.set_shapes([channel_dim, None, None], output_shape)

        input_dim = self.embed.get_output_dim(-3)
        layer = InputReluCnnLayer(input_dim, channel_dim, kernel_size)
        self.layers.append(layer)

        for b in blocks[:-1]:
            for _ in range(b - 1):
                layer = ReluCnnLayer(channel_dim, kernel_size)
                self.layers.append(layer)

            layer = StridedReluCnnLayer(channel_dim, kernel_size, stride)
            self.layers.append(layer)

        for _ in range(blocks[-1]):
            layer = ReluCnnLayer(channel_dim, kernel_size)
            self.layers.append(layer)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("hidden_dim",           type=int, default=100),
            cli.Arg("num_hidden_layers",    type=int, default=3),
        ]

class ReluCnnLayer(Model):
    def __init__(
        self,
        channel_dim: int,
        kernel_size: int,
    ):
        self._torch_module_init()
        self.conv = torch.nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=kernel_size,
            padding="same",
        )

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        x_nchw = self.conv.forward(x_nchw)
        x_nchw = torch.relu(x_nchw)
        return x_nchw

class InputReluCnnLayer(ReluCnnLayer):
    def __init__(
        self,
        input_channel_dim: int,
        output_channel_dim: int,
        kernel_size: int,
    ):
        self._torch_module_init()
        self.conv = torch.nn.Conv2d(
            in_channels=input_channel_dim,
            out_channels=output_channel_dim,
            kernel_size=kernel_size,
            padding="same",
        )

class StridedReluCnnLayer(ReluCnnLayer):
    def __init__(
        self,
        channel_dim: int,
        kernel_size: int,
        stride: int,
    ):
        self._torch_module_init()
        self.conv = torch.nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
