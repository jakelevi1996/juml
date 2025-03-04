import torch
import torch.utils.data
from jutility import util
import juml
import juml_test_utils

OUTPUT_DIR = juml_test_utils.get_output_dir("test_datasets")

def test_linear():
    printer = util.Printer("test_linear", dir_name=OUTPUT_DIR)
    juml_test_utils.set_torch_seed("test_linear")

    input_dim   = 7
    output_dim  = 11
    n_train     = 23
    n_test      = 27
    batch_size  = 17

    dataset = juml.datasets.Linear(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
        n_test=n_test,
        x_std=0.1,
        t_std=0.2,
    )
    assert repr(dataset) == "Linear(n_train=23, n_test=27)"
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

    assert isinstance(dataset.loss, juml.datasets.loss.Loss)
    assert isinstance(dataset.loss, juml.datasets.loss.Mse)

def test_mnist():
    printer = util.Printer("test_mnist", dir_name=OUTPUT_DIR)
    juml_test_utils.set_torch_seed("test_mnist")

    dataset = juml.datasets.Mnist()
    assert repr(dataset) == "Mnist(n_train=60.0k, n_test=10.0k)"
    assert dataset.get_input_shape()  == [1, 28, 28]
    assert dataset.get_output_shape() == [10]

    train_split = dataset.get_data_split("train")
    assert isinstance(train_split, torch.utils.data.Dataset)
    assert len(train_split) == 60000

    test_split = dataset.get_data_split("test")
    assert isinstance(test_split, torch.utils.data.Dataset)
    assert len(test_split) == 10000

    for batch_size in [64, 100, 128]:
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
