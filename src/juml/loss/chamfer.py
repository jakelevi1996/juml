import torch
from jutility import cli
from juml.loss.base import Loss

class ChamferMse(Loss):
    def chamfer_components(self, y_bchw: torch.Tensor, t_bpc: torch.Tensor):
        y_bcq   = y_bchw.flatten(-2, -1)
        y_b1cq  = y_bcq.unsqueeze(-3)
        t_bpc1  = t_bpc.unsqueeze(-1)
        e_bpcq  = self.weights * (y_b1cq - t_bpc1)
        mse_bpq = e_bpcq.square().sum(dim=-2)
        mse_bp  = mse_bpq.min(dim=-1).values
        mse_bq  = mse_bpq.min(dim=-2).values
        return mse_bp, mse_bq

    def forward(self, y, t):
        mse_bp, mse_bq = self.chamfer_components(y, t)
        return mse_bp.mean() + mse_bq.mean()

    def info(self):
        return {"ylabel": "Chamfer loss", "log_y": True}

    def metric_batch(self, y, t):
        mse_bp, mse_bq = self.chamfer_components(y, t)
        return (
            mse_bp.mean(dim=-1).sum().item() +
            mse_bq.mean(dim=-1).sum().item()
        )

    def metric_info(self):
        return {"ylabel": "Chamfer MSE", "log_y": True}

    def needs_weights(self) -> bool:
        return True

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(cls, tag="CH")
