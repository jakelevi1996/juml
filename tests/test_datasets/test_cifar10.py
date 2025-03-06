import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_datasets/test_cifar10")

def test_cifar10():
    printer = util.Printer("test_cifar10", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_cifar10")

    dataset = juml.datasets.Cifar10()
    assert dataset.get_input_shape()  == [3, 32, 32]
    assert dataset.get_output_shape() == [10]

    batch_size = 64
    data_loader = dataset.get_data_loader("train", batch_size)
    x, t = next(iter(data_loader))

    assert isinstance(x, torch.Tensor)
    assert list(x.shape) == [batch_size, 3, 32, 32]
    assert x.dtype is torch.float32
    assert x.dtype is not torch.int64
    assert x.max().item() == 1
    assert x.min().item() == 0

    assert isinstance(t, torch.Tensor)
    assert list(t.shape) == [batch_size]
    assert t.dtype is torch.int64
    assert t.dtype is not torch.float32
    assert t.max().item() == 9
    assert t.min().item() == 0
