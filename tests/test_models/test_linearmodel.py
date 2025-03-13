import math
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_linearmodel")

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
        pooler=juml.models.pool.Identity(),
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
        embedder=juml.models.embed.Flatten(n=3),
        pooler=juml.models.pool.Identity(),
    )
    assert repr(model) == "LinearModel(num_params=1.6k)"
    assert model.num_params() == (4 * 5 * input_dim * output_dim) + output_dim

    y = model.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, output_dim]

def test_linearmodel_unflatten():
    printer = util.Printer("test_linearmodel_unflatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_linearmodel_unflatten")

    x = torch.rand([3, 4, 5, 6])
    t = torch.rand([3, 4, 5, 7])

    pooler = juml.models.pool.Unflatten(2)
    embedder = juml.models.embed.Flatten(2)
    model = juml.models.LinearModel(
        input_shape=x.shape,
        output_shape=t.shape,
        embedder=embedder,
        pooler=pooler,
    )
    assert pooler.get_input_shape() == [3, 4, (5 * 7)]
    assert pooler.get_input_dim(-1) == (5 * 7)
    assert repr(embedder)       == "Flatten(num_params=0)"
    assert repr(pooler)         == "Unflatten(num_params=0)"
    assert repr(model)          == "LinearModel(num_params=1.1k)"

    [layer] = model.layers
    assert isinstance(layer, juml.models.Linear)
    assert repr(layer) == "Linear(num_params=1.1k)"
    assert list(layer.w_io.shape) == [30, 35]
    assert list(layer.b_o.shape)  == [35]

    y = model.forward(x)
    assert list(y.shape) == [3, 4, 5, 7]
