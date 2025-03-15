import os
import torch
from jutility import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def torch_set_print_options(
    precision:  int=3,
    linewidth:  int=10000,
    sci_mode:   bool=False,
    threshold:  (int | float)=1e9,
):
    torch.set_printoptions(
        precision=precision,
        linewidth=linewidth,
        sci_mode=sci_mode,
        threshold=int(threshold),
    )

def get_output_dir(*subdir_names: str):
    return os.path.join("tests", "Outputs", *subdir_names)

def set_torch_seed(*args):
    seed = util.Seeder().get_seed(*args)
    torch.manual_seed(seed)
