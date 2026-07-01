import torch
from juml.models.model import Model

class LinearLayer(Model):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        w_scale:    (float | None)=None,
    ):
        if w_scale is None:
            w_scale = input_dim ** (-1/2)

        self._torch_module_init()
        w_shape = [input_dim, output_dim]
        self.w_io = torch.nn.Parameter(torch.normal(0, w_scale, w_shape))
        self.b_o = torch.nn.Parameter(torch.zeros([output_dim]))

    def forward(self, x_ni: torch.Tensor) -> torch.Tensor:
        y_no = x_ni @ self.w_io + self.b_o
        return y_no
