import torch
import torch.utils.data
from juml.device import device

class Loss(torch.nn.Module):
    def _torch_module_init(self):
        super().__init__()

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def info(self) -> dict:
        raise NotImplementedError()

    def metric_batch(self, y: torch.Tensor, t: torch.Tensor) -> float:
        raise NotImplementedError()

    def metric_info(self) -> dict:
        raise NotImplementedError()

    def metric(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        gpu: bool,
    ) -> float:
        metric_sum = 0
        for x, t in data_loader:
            x, t = device([x, t], gpu)
            y = model.forward(x)
            metric_sum += self.metric_batch(y, t)

        return metric_sum / len(data_loader.dataset)

class CrossEntropy(Loss):
    def forward(self, y, t):
        return torch.nn.functional.cross_entropy(y, t)

    def info(self):
        return {"ylabel": "Loss"}

    def metric_batch(self, y, t):
        return torch.where(y.argmax(dim=-1) == t, 1, 0).sum().item()

    def metric_info(self):
        return {"ylabel": "Accuracy", "ylim": [0, 1]}

class Mse(Loss):
    def forward(self, y, t):
        return (y - t).square().sum(dim=-1).mean()

    def info(self):
        return {"ylabel": "Loss", "log_y": True}

    def metric_batch(self, y, t):
        return (y - t).square().sum().item()

    def metric_info(self):
        return {"ylabel": "MSE", "log_y": True}

class WeightedSetMse(Mse):
    def __init__(self, weights: torch.Tensor):
        self._torch_module_init()
        self.weights = torch.nn.Parameter(
            weights.flatten(),
            requires_grad=False,
        )

    def forward(self, y_bpo: torch.Tensor, t_bpo: torch.Tensor):
        e_bpo = self.weights * (y_bpo - t_bpo)
        return e_bpo.square().sum(dim=-1).mean()

    def metric_batch(self, y_bpo: torch.Tensor, t_bpo: torch.Tensor):
        e_bpo = self.weights * (y_bpo - t_bpo)
        return e_bpo.square().mean(dim=-2).sum().item()

class ChamferMse(Loss):
    def __init__(self, weights: torch.Tensor):
        self._torch_module_init()
        self.weights = torch.nn.Parameter(
            weights.flatten().unsqueeze(-1),
            requires_grad=False,
        )

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
