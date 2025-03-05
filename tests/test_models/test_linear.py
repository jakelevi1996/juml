import math
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_linear")

def test_linearmodel():
    printer = util.Printer("test_linearmodel", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_linearmodel")

    input_dim  = 7
    output_dim = 11
    x = torch.rand([3, 4, 5, input_dim])

    model = juml.models.LinearModel(
        input_shape=list(x.shape),
        output_shape=[output_dim],
        embedder=juml.models.embed.Identity(),
    )
    assert repr(model) == "LinearModel(num_params=88)"
    assert model.num_params() == (input_dim * output_dim) + output_dim

    y = model.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, 4, 5, output_dim]

def test_linearmodel_flatten():
    printer = util.Printer("test_linearmodel_flatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_linearmodel_flatten")

    input_dim  = 7
    output_dim = 11
    x = torch.rand([3, 4, 5, input_dim])

    model = juml.models.LinearModel(
        input_shape=list(x.shape),
        output_shape=[output_dim],
        embedder=juml.models.embed.Flatten(num_flatten=3),
    )
    assert repr(model) == "LinearModel(num_params=1.6k)"
    assert model.num_params() == (4 * 5 * input_dim * output_dim) + output_dim

    y = model.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, output_dim]
