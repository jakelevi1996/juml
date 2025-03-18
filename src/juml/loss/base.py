import torch
import torch.utils.data
from jutility import util, cli
from juml import device

class Loss:
    def __init__(self):
        self.weights = None

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def info(self) -> dict:
        raise NotImplementedError()

    def metric_batch(self, y: torch.Tensor, t: torch.Tensor) -> float:
        raise NotImplementedError()

    def metric_info(self) -> dict:
        return self.info()

    def metric(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        gpu: bool,
    ) -> float:
        metric_sum = 0
        for x, t in data_loader:
            x, t = device.to_device([x, t], gpu)
            y = model.forward(x)
            metric_sum += self.metric_batch(y, t)

        return metric_sum / len(data_loader.dataset)

    def needs_weights(self) -> bool:
        return False

    def set_weights(self, weights: torch.Tensor):
        self.weights = weights

    def cuda(self):
        if self.weights is not None:
            self.weights = self.weights.cuda()

    @classmethod
    def get_cli_arg(cls):
        return cli.ObjectArg(cls)

    def __repr__(self) -> str:
        return util.format_type(type(self), weights=self.weights)
