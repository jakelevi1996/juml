import torch
from jutility import cli
from juml.models.model import Model

class ClassificationModel(Model):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def step(self, x: torch.Tensor, t: torch.Tensor) -> float:
        raise NotImplementedError()
