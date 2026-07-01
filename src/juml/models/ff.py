import torch
from jutility import cli
from juml.models.model import Model

class FeedForwardModel(Model):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
