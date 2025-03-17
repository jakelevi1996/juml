import pytest
import torch
from jutility import util
import juml
import juml.models.rzcnn

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_pool")

def test_unflatten():
    printer = util.Printer("test_unflatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_unflatten")

    pooler = juml.models.pool.Unflatten(n=2)

    x = torch.rand([3, 4, 30])

    pooler.set_shapes([], [4, 5, 6])
    assert repr(pooler) == "Unflatten(num_params=0)"
    assert pooler.num_params() == 0
    assert pooler.get_input_shape() == [4, 30]
    assert pooler.get_input_dim(-1) == 30

    y = pooler.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, 4, 5, 6]
    printer(y.max(), y.min())
    assert y.max().item() <= 1
    assert y.min().item() >= 0

    with pytest.raises(RuntimeError):
        assert torch.all(y == x)

    assert torch.all(y.flatten(-2, -1) == x)

def test_unflatten_model():
    printer = util.Printer("test_unflatten_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_unflatten_model")

    pooler = juml.models.pool.Unflatten(n=2)

    x = torch.rand([3, 4, 5])
    t = torch.rand([3, 4, 6, 7])

    model = juml.models.Mlp(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        hidden_dim=13,
        num_hidden_layers=2,
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
