import torch
import torch.utils.data
from jutility import util
from juml.data.classification import ClassificationDataset
from juml.models.ff import FeedForwardModel
from juml.device import DeviceConfig

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
    return (
        + t * torch.nn.functional.softplus(-y)
        + (1 - t) * torch.nn.functional.softplus(y)
    )

def binary_cross_entropy_from_probs(
    y: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return -torch.xlogy(t, y) - torch.xlogy(1 - t, 1 - y)

def binary_acc(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    y_hard = torch.where(y > 0.5, 1, 0)
    acc = torch.where(t == y_hard, 1.0, 0.0).mean()
    return acc

def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(b != 0.0, a / b, 0.0)

def reg_lstsq(
    a_ni:   torch.Tensor,
    b_no:   torch.Tensor,
    reg:    float,
) -> torch.Tensor:
    ab_io = a_ni.mT @ b_no
    aa_ii = a_ni.mT @ a_ni
    aa_reg_ii = aa_ii + reg * torch.eye(*aa_ii.shape)
    x_io = torch.linalg.solve(aa_reg_ii, ab_io)
    return x_io

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

def use_float64():
    torch.set_default_dtype(torch.float64)

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

def batched_multiclass_acc(
    model:          FeedForwardModel,
    data_loader:    torch.utils.data.DataLoader,
    dataset:        ClassificationDataset,
    device_cfg:     DeviceConfig,
) -> float:
    n_correct = 0
    n_total   = 0
    for x, t in data_loader:
        x, t = device_cfg.set_tensor_device(x, t)
        x, t = dataset.format_batch(x, t)
        y = model.forward(x)
        acc_bool = y.argmax(-1) == t.argmax(-1)
        n_correct += torch.where(acc_bool, 1, 0).sum().item()
        n_total   += acc_bool.numel()

    return n_correct / n_total
