import pytest
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_datasets/test_mnist")

def test_mnist():
    printer = util.Printer("test_mnist", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mnist")

    dataset = juml.datasets.Mnist()
    assert repr(dataset) == "Mnist(n_train=60.0k, n_test=10.0k)"
    assert dataset.get_input_shape()  == [1, 28, 28]
    assert dataset.get_output_shape() == [10]
    assert dataset.get_default_loss() == "CrossEntropy"

    train_split = dataset.get_data_split("train")
    assert isinstance(train_split, torch.utils.data.Dataset)
    assert len(train_split) == 60000

    test_split = dataset.get_data_split("test")
    assert isinstance(test_split, torch.utils.data.Dataset)
    assert len(test_split) == 10000

    batch_size = 64
    data_loader = dataset.get_data_loader("train", batch_size)
    x, t = next(iter(data_loader))

    assert isinstance(x, torch.Tensor)
    assert list(x.shape) == [batch_size, 1, 28, 28]
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

    with pytest.raises(NotImplementedError):
        dataset.get_loss_weights()
