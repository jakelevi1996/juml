import os
import torch

def to_device(args: list[torch.Tensor], gpu: bool) -> list[torch.Tensor]:
    if gpu:
        args = [x.cuda() for x in args]

    return args

def set_visible(devices: list[int]):
    devices_str = ",".join(str(d) for d in devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
