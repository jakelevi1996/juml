import math
import torch

ZERO = torch.tensor(0.0)

def softmax_cross_entropy_from_logits(
    y:      torch.Tensor,
    t:      torch.Tensor,
    dim:    int,
) -> torch.Tensor:
    return -(t * y).sum(dim) + y.logsumexp(dim)

def softmax_cross_entropy_from_probs(
    y:      torch.Tensor,
    t:      torch.Tensor,
    dim:    int,
) -> torch.Tensor:
    return -torch.xlogy(t, y).sum(dim)

def multiclass_acc(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    y_hard = y.argmax(-1)
    t_hard = t.argmax(-1)
    acc = torch.where(t_hard == y_hard, 1.0, 0.0).mean()
    return acc

def binary_cross_entropy_from_logits(
    y: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return t * softplus(-y) + (1 - t) * softplus(y)

def binary_cross_entropy_from_probs(
    y: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return -torch.xlogy(t, y) - torch.xlogy(1 - t, 1 - y)

def binary_acc(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    y_hard = torch.where(y > 0.5, 1, 0)
    acc = torch.where(t == y_hard, 1.0, 0.0).mean()
    return acc

def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(ZERO, x)

def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(b != 0.0, a / b, 0.0)

def error_if_not_finite(x: torch.Tensor):
    if not x.isfinite().all():
        raise RuntimeError()
