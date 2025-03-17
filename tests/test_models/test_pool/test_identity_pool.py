import torch
from jutility import util
import juml
import juml.models.rzcnn

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_pool")

def test_identity():
    printer = util.Printer("test_identity", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_identity")

    pooler = juml.models.pool.Identity()

    x = torch.rand([3, 4, 5, 6])

    pooler.set_shapes([None], list(x.shape))
    assert repr(pooler) == "Identity(num_params=0)"
    assert pooler.num_params() == 0
    assert pooler.get_input_shape() == list(x.shape)
    assert pooler.get_input_dim(-1) == 6

    y = pooler.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == list(x.shape)
    printer(y.max(), y.min())
    assert y.max().item() <= 2
    assert y.min().item() >= -2

    assert torch.all(y == x)

def test_identity_model():
    printer = util.Printer("test_identity_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_identity_model")

    pooler = juml.models.pool.Identity()

    x = torch.rand([3, 4, 5, 6])
    t = torch.rand([3, 4, 5, 6])

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
