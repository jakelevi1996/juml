import math
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_datasets/test_dataset")

def test_get_data_loader():
    printer = util.Printer("test_get_data_loader", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_get_data_loader")

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
    juml.test_utils.set_torch_seed("test_get_subset_loader")

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

    x_subset    = set(tuple(xi) for xi in x_subset.tolist())
    t_subset    = set(tuple(ti) for ti in t_subset.tolist())
    x_full      = set(tuple(xi) for xi in x_full.tolist())
    t_full      = set(tuple(ti) for ti in t_full.tolist())
    assert x_subset < x_full
    assert t_subset < t_full
    assert not x_full < x_subset
    assert not t_full < t_subset
    assert len(x_subset)    == n_subset
    assert len(t_subset)    == n_subset
    assert len(x_full)      == n_train
    assert len(t_full)      == n_train
