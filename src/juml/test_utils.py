import os
import torch
from jutility import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def torch_set_print_options(
    precision=3,
    linewidth=10000,
    sci_mode=False,
    threshold=int(1e9),
):
    torch.set_printoptions(
        precision=precision,
        linewidth=linewidth,
        sci_mode=sci_mode,
        threshold=threshold,
    )

def get_output_dir(*subdir_names):
    return os.path.join("tests", "Outputs", *subdir_names)

def set_torch_seed(*args):
    seed = util.Seeder().get_seed(*args)
    torch.manual_seed(seed)
