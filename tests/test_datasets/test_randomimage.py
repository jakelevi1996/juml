import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_datasets/test_randomimage")

def test_randomimage():
    printer = util.Printer("test_randomimage", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_randomimage")

    input_shape = [7, 26, 32]
    num_classes = 13
    num_train   = 123
    num_test    = 456
    dataset = juml.datasets.RandomImage(
        input_shape=input_shape,
        num_classes=num_classes,
        train=num_train,
        test=num_test,
    )
    assert repr(dataset) == "RandomImage(n_train=123, n_test=456)"
    assert dataset.get_input_shape()    == input_shape
    assert dataset.get_output_shape()   == [num_classes]

    train_split = dataset.get_data_split("train")
    assert isinstance(train_split, torch.utils.data.Dataset)
    assert len(train_split) == num_train

    test_split = dataset.get_data_split("test")
    assert isinstance(test_split, torch.utils.data.Dataset)
    assert len(test_split) == num_test

    batch_size = 17
    data_loader = dataset.get_data_loader("train", batch_size)
    x, t = next(iter(data_loader))

    assert isinstance(x, torch.Tensor)
    assert x.dtype is torch.float32
    assert x.dtype is not torch.int64
    assert list(x.shape) == [batch_size, *input_shape]
    printer(x.max(), x.min())
    assert x.max().item() <= 1
    assert x.min().item() >= 0

    assert isinstance(t, torch.Tensor)
    assert t.dtype is torch.int64
    assert t.dtype is not torch.float32
    assert list(t.shape) == [batch_size]
    printer(t.max(), t.min())
    assert t.max().item() == num_classes - 1
    assert t.min().item() == 0
