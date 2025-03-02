import torch

def device(args: list[torch.Tensor], gpu: bool) -> list[torch.Tensor]:
    if gpu:
        args = [x.cuda() for x in args]

    return args
