import torch

def to_device(args: list[torch.Tensor], gpu: bool) -> list[torch.Tensor]:
    if gpu:
        args = [x.cuda() for x in args]

    return args
