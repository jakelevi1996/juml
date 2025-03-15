import torch
from juml.loss.base import Loss

class CrossEntropy(Loss):
    def forward(self, y, t):
        return torch.nn.functional.cross_entropy(y, t)

    def info(self):
        return {"ylabel": "Loss"}

    def metric_batch(self, y, t):
        return torch.where(y.argmax(dim=-1) == t, 1, 0).sum().item()

    def metric_info(self):
        return {"ylabel": "Accuracy", "ylim": [0, 1]}
