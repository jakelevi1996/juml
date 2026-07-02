import torch
from jutility import util

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

def all_in_range(x: torch.Tensor, x_lo: float, x_hi: float) -> bool:
    return (x >= x_lo).all().item() and (x <= x_hi).all().item()

def all_close_to_zero(x: torch.Tensor, tol: float=1e-5) -> bool:
    return all_in_range(x, -tol, tol)

def all_close(x: torch.Tensor, y: torch.Tensor, tol: float=1e-5) -> bool:
    return all_close_to_zero(x - y, tol)

def set_torch_seed(*args):
    seed = util.Seeder().get_seed(*args)
    torch.manual_seed(seed)

def torch_set_print_options(
    precision:  int=3,
    threshold:  (int | float)=1e3,
    linewidth:  (int | float)=1e5,
    sci_mode:   bool=False,
):
    torch.set_printoptions(
        precision=precision,
        threshold=int(threshold),
        linewidth=int(linewidth),
        sci_mode=sci_mode,
    )

class TensorPrinter:
    def __init__(self, printer: (util.Printer | None)=None):
        if printer is None:
            printer = util.Printer()

        self.printer = printer

    @classmethod
    def format(self, x: torch.Tensor) -> str:
        parts = [
            "shape = %s" % list(x.shape),
            "numel = %s" % x.numel(),
            "dtype = %s" % x.dtype,
            str(x),
        ]
        return "\n".join(parts)

    def __call__(self, x: torch.Tensor, name: (str | None)=None):
        if name is not None:
            self.printer((" %s " % name).center(util.HLINE_LEN, "-"))
        else:
            self.printer.hline()

        self.printer(self.format(x))
        self.printer.hline()
