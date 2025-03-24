import math
import torch
from juml.models.base import Model

TOL = 1e-5

class Linear(Model):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        w_scale:    (float | None)=None,
    ):
        if w_scale is None:
            w_scale = 1 / math.sqrt(input_dim)

        w_shape = [input_dim, output_dim]
        self._torch_module_init()
        self.w_io = torch.nn.Parameter(torch.normal(0, w_scale, w_shape))
        self.b_o  = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x_ni: torch.Tensor) -> torch.Tensor:
        x_no = x_ni @ self.w_io + self.b_o
        return x_no

    def init_batch(self, x: torch.Tensor, t: (torch.Tensor | None)):
        with torch.no_grad():
            self.b_o.zero_()
            if t is None:
                x = x.flatten(0, -2)
                self.w_io *= 1 / (TOL + x.std(-2).unsqueeze(-1))
                y = self.forward(x)
                self.w_io *= 1 / (TOL + y.std(-2).unsqueeze(-2))
                y = self.forward(x)
                self.b_o.copy_(-y.mean(-2))
            else:
                x = x.flatten(0, -2)
                t = t.flatten(0, -2)
                xm = x.mean(-2, keepdim=True)
                tm = t.mean(-2, keepdim=True)
                xc = x - xm
                tc = t - tm
                cov_xt_io = xc.T @ tc
                cov_xx_ii = xc.T @ xc
                cov_xx_ii.diagonal().add_(TOL)
                self.w_io.copy_(torch.linalg.solve(cov_xx_ii, cov_xt_io))
                self.b_o.copy_((tm - self.forward(xm)).squeeze(-2))

class MultiHeadLinear(Model):
    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        num_heads:  int,
        w_scale:    float,
    ):
        self._torch_module_init()
        w_shape = [num_heads, input_dim, output_dim]
        b_shape = [num_heads, 1,         output_dim]
        self.w_hio = torch.nn.Parameter(torch.normal(0, w_scale, w_shape))
        self.b_h1o = torch.nn.Parameter(torch.zeros(b_shape))

    def forward(self, x_n1pi: torch.Tensor) -> torch.Tensor:
        x_nhpo = x_n1pi @ self.w_hio + self.b_h1o
        return x_nhpo
