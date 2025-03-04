import math
import torch
import torch.utils.data
from jutility import util
import juml
import juml_test_utils

OUTPUT_DIR = juml_test_utils.get_output_dir("test_datasets")

def test_get_data_loader():
    printer = util.Printer("test_get_data_loader", dir_name=OUTPUT_DIR)
    juml_test_utils.set_torch_seed("test_get_data_loader")

    input_dim   = 7
    output_dim  = 11
    n_train     = 200
    n_test      = 100

    dataset = juml.datasets.Linear(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
        n_test=n_test,
        x_std=0.1,
        t_std=0.2,
    )

    for batch_size in [64, 100, 128]:
        data_loader = dataset.get_data_loader("train", batch_size)
        assert isinstance(data_loader, torch.utils.data.DataLoader)

        x, t = next(iter(data_loader))

        assert isinstance(x, torch.Tensor)
        assert list(x.shape) == [batch_size, input_dim]
        assert x.dtype is torch.float32
        assert x.dtype is not torch.int64

        assert isinstance(t, torch.Tensor)
        assert list(t.shape) == [batch_size, output_dim]
        assert t.dtype is torch.float32
        assert t.dtype is not torch.int64

    batch_size = 17
    batch_size_list = []

    data_loader = dataset.get_data_loader("train", batch_size)
    assert isinstance(data_loader, torch.utils.data.DataLoader)

    for x, t in data_loader:
        assert isinstance(x, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        assert x.shape[0 ] == t.shape[0 ]
        assert x.shape[1:] != t.shape[1:]

        batch_size_list.append(x.shape[0])

    assert sum(batch_size_list) == n_train
    assert set(batch_size_list) == set([batch_size, n_train % batch_size])
    assert len(batch_size_list) == math.ceil(n_train / batch_size)
    assert len(set(batch_size_list)) == 2

    printer(batch_size_list)

def test_get_subset_loader():
    printer = util.Printer("test_get_subset_loader", dir_name=OUTPUT_DIR)
    juml_test_utils.set_torch_seed("test_get_subset_loader")

    input_dim   = 7
    output_dim  = 11
    n_train     = 200
    n_test      = 100

    dataset = juml.datasets.Linear(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
        n_test=n_test,
        x_std=0.1,
        t_std=0.2,
    )

    n_subset = 37
    batch_size = 17
    batch_size_list = []

    data_loader = dataset.get_subset_loader("train", n_subset, batch_size)
    assert isinstance(data_loader, torch.utils.data.DataLoader)

    for x, t in data_loader:
        assert isinstance(x, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        assert x.shape[0 ] == t.shape[0 ]
        assert x.shape[1:] != t.shape[1:]

        batch_size_list.append(x.shape[0])

    assert sum(batch_size_list) == n_subset
    assert sum(batch_size_list) <  n_train
    assert set(batch_size_list) == set([batch_size, n_subset % batch_size])
    assert len(batch_size_list) == math.ceil(n_subset / batch_size)

    printer(batch_size_list)

    full_dl     = dataset.get_data_loader(  "train", n_train)
    subset_dl   = dataset.get_subset_loader("train", n_subset, n_subset)
    x_full,     t_full      = next(iter(full_dl))
    x_subset,   t_subset    = next(iter(subset_dl))
    assert isinstance(x_full,   torch.Tensor)
    assert isinstance(t_full,   torch.Tensor)
    assert isinstance(x_subset, torch.Tensor)
    assert isinstance(t_subset, torch.Tensor)
    assert list(x_full.shape)   == [n_train,    input_dim]
    assert list(t_full.shape)   == [n_train,    output_dim]
    assert list(x_subset.shape) == [n_subset,   input_dim]
    assert list(t_subset.shape) == [n_subset,   output_dim]
    assert (
        set(tuple(xi) for xi in x_subset.tolist()) <
        set(tuple(xi) for xi in x_full.tolist())
    )
    assert (
        set(tuple(ti) for ti in t_subset.tolist()) <
        set(tuple(ti) for ti in t_full.tolist())
    )

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
