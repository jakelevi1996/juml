import pytest
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_datasets/test_linear")

def test_linear():
    printer = util.Printer("test_linear", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_linear")

    input_dim   = 7
    output_dim  = 11
    n_train     = 23
    n_test      = 27
    batch_size  = 17

    dataset = juml.datasets.LinearDataset(
        input_dim=input_dim,
        output_dim=output_dim,
        train=n_train,
        test=n_test,
        x_std=0.1,
        t_std=0.2,
    )
    assert repr(dataset) == "LinearDataset(n_train=23, n_test=27)"
    assert dataset.get_input_shape()  == [input_dim ]
    assert dataset.get_output_shape() == [output_dim]

    train_split = dataset.get_data_split("train")
    assert isinstance(train_split, torch.utils.data.Dataset)
    assert len(train_split) == n_train

    test_split = dataset.get_data_split("test")
    assert isinstance(test_split, torch.utils.data.Dataset)
    assert len(test_split) == n_test

    data_loader = dataset.get_data_loader("train", batch_size)
    x, t = next(iter(data_loader))

    assert isinstance(x, torch.Tensor)
    assert list(x.shape) == [batch_size, input_dim]
    assert x.dtype is torch.float32
    assert x.dtype is not torch.int64
    assert x.max().item() > 0
    assert x.min().item() < 0

    assert isinstance(t, torch.Tensor)
    assert list(t.shape) == [batch_size, output_dim]
    assert t.dtype is torch.float32
    assert t.dtype is not torch.int64
    assert t.max().item() > 0
    assert t.min().item() < 0

    assert dataset.get_default_loss() == "Mse"
    assert isinstance(dataset.get_loss_weights(), torch.Tensor)
    assert list(dataset.get_loss_weights().shape) == [output_dim]
