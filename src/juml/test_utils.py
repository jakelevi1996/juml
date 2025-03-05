import os
import torch
from jutility import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

util.numpy_set_print_options()

torch.set_printoptions(
    precision=3,
    linewidth=10000,
    sci_mode=False,
    threshold=int(1e9),
)

def get_output_dir(*subdir_names):
    return os.path.join("tests", "Outputs", *subdir_names)

def set_torch_seed(*args):
    seed = util.Seeder().get_seed(*args)
    torch.manual_seed(seed)
