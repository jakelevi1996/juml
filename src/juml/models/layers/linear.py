import torch
from juml.models.model import Model
from juml.util import reg_lstsq

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

    def normalise(self, x_ni: torch.Tensor, b_std: float=1.0):
        y_no = x_ni @ self.w_io
        s_1o = y_no.std(0, keepdim=True)
        with torch.no_grad():
            self.w_io /= s_1o

        y_no = x_ni @ self.w_io
        m_o = y_no.mean(0, keepdim=False)
        with torch.no_grad():
            self.b_o.copy_((b_std * torch.randn_like(self.b_o)) - m_o)

    def lstsq(self, x_ni: torch.Tensor, t_no: torch.Tensor, reg: float):
        xm_i = x_ni.mean(0)
        tm_o = t_no.mean(0)
        xc_ni = x_ni - xm_i
        tc_no = t_no - tm_o

        with torch.no_grad():
            self.w_io.copy_(reg_lstsq(xc_ni, tc_no, reg))
            self.b_o.copy_(tm_o - xm_i @ self.w_io)

    def indirect_lstsq(
        self,
        x_ni:   torch.Tensor,
        t_nd:   torch.Tensor,
        A_od:   torch.Tensor,
        d_nd:   torch.Tensor,
        a_reg:  float,
        x_reg:  float,
    ):
        td_nd = t_nd - d_nd

        xm_i = x_ni.mean(0)
        tm_d = td_nd.mean(0)
        xc_ni = x_ni - xm_i
        tc_nd = td_nd - tm_d

        atm_o = reg_lstsq(A_od.mT, tm_d, a_reg)
        atc_no = reg_lstsq(A_od.mT, tc_nd.mT, a_reg).mT

        with torch.no_grad():
            self.w_io.copy_(reg_lstsq(xc_ni, atc_no, x_reg))
            self.b_o.copy_(atm_o - xm_i @ self.w_io)
