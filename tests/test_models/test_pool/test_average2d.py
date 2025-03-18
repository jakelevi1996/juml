import pytest
import torch
from jutility import util
import juml
import juml.models.rzcnn

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_pool")

def test_average2d():
    printer = util.Printer("test_average2d", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_average2d")

    pooler = juml.models.pool.Average2d()

    x = torch.rand([3, 4, 5, 6])

    pooler.set_shapes(list(x.shape), [11])
    assert repr(pooler) == "Average2d(num_params=55)"
    assert pooler.num_params() == (4*11 + 11)
    assert pooler.num_params() == 55
    with pytest.raises(NotImplementedError):
        pooler.get_input_shape()
    with pytest.raises(NotImplementedError):
        pooler.get_input_dim(-1)

    y = pooler.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, 11]
    printer(y.max(), y.min())
    assert y.max().item() <= 2
    assert y.min().item() >= -2

def test_average2d_model():
    printer = util.Printer("test_average2d_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_average2d_model")

    pooler = juml.models.pool.Average2d()

    x = torch.rand([3, 4, 11, 13])
    t = torch.rand([3, 7])

    model = juml.models.RzCnn(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        kernel_size=3,
        model_dim=9,
        expand_ratio=2.0,
        num_stages=2,
        blocks_per_stage=3,
        stride=2,
        embedder=juml.models.embed.Identity(),
        pooler=pooler,
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())
    assert model.pool is pooler

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert y_0.dtype is torch.float32
    assert y_0.dtype is not torch.int64
    assert list(y_0.shape) == list(t.shape)
    printer(y_0.max(), y_0.min())
    assert y_0.max().item() <= 2
    assert y_0.min().item() >= -2

    loss_0 = loss.forward(y_0, t)
    loss_0.backward()
    optimiser.step()

    y_1 = model.forward(x)
    loss_1 = loss.forward(y_1, t)

    printer(loss_0, loss_1)
    assert loss_1.item() < loss_0.item()
