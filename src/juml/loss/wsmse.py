import torch
from juml.loss.mse import Mse

class WeightedSetMse(Mse):
    def forward(self, y_bpo: torch.Tensor, t_bpo: torch.Tensor):
        e_bpo = self.weights * (y_bpo - t_bpo)
        return e_bpo.square().sum(dim=-1).mean()

    def metric_batch(self, y_bpo: torch.Tensor, t_bpo: torch.Tensor):
        e_bpo = self.weights * (y_bpo - t_bpo)
        return e_bpo.square().mean(dim=-2).sum().item()

    def needs_weights(self) -> bool:
        return True
