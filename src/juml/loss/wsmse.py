import torch
from juml.loss.mse import Mse

class WeightedSetMse(Mse):
    def forward(self, y_npo: torch.Tensor, t_npo: torch.Tensor):
        e_npo = self.weights * (y_npo - t_npo)
        return e_npo.square().sum(dim=-1).mean()

    def metric_batch(self, y_npo: torch.Tensor, t_npo: torch.Tensor):
        e_npo = self.weights * (y_npo - t_npo)
        return e_npo.square().mean(dim=-2).sum().item()

    def needs_weights(self) -> bool:
        return True
