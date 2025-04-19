import os
import torch
from jutility import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def get_output_dir(*subdir_names: str):
    return os.path.join("tests", "Outputs", *subdir_names)

def set_torch_seed(*args):
    seed = util.Seeder().get_seed(*args)
    torch.manual_seed(seed)

class TensorPrinter:
    def __init__(self, printer: util.Printer):
        self.printer = printer

    def __call__(self, x: torch.Tensor):
        self.printer(x.numel(), x.shape, x, sep="\n")
        self.printer.hline()

class TensorPrinterMarkdown(TensorPrinter):
    def __call__(self, heading: str, x: torch.Tensor):
        self.printer(
            "\n## %s\n\n```txt\nnumel=%s\nshape=%s\n%s\n```"
            % (heading, x.numel(), x.shape, x)
        )
