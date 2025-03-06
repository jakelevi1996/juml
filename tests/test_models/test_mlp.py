import math
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_mlp")

def test_mlp():
    printer = util.Printer("test_mlp", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mlp")

    input_dim  = 7
    output_dim = 11
    x = torch.rand([3, 4, 5, input_dim])

    model = juml.models.Mlp(
        input_shape=list(x.shape),
        output_shape=[output_dim],
        hidden_dim=13,
        num_hidden_layers=2,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    assert repr(model) == "Mlp(num_params=440)"
    assert model.num_params() == 440

    y = model.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, 4, 5, output_dim]

def test_mlp_flatten():
    printer = util.Printer("test_mlp_flatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mlp_flatten")

    input_dim  = 7
    output_dim = 11
    x = torch.rand([3, 4, 5, input_dim])

    model = juml.models.Mlp(
        input_shape=list(x.shape),
        output_shape=[output_dim],
        hidden_dim=13,
        num_hidden_layers=2,
        embedder=juml.models.embed.Flatten(num_flatten=3),
        pooler=juml.models.pool.Identity(),
    )
    assert repr(model) == "Mlp(num_params=2.2k)"
    assert model.num_params() == 2169

    y = model.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, output_dim]
