import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_embed")

def test_identity():
    printer = util.Printer("test_identity", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_identity")

    embedder = juml.models.embed.Identity()

    x = torch.rand([3, 4, 5, 6])

    embedder.set_input_shape(list(x.shape))
    assert repr(embedder) == "Identity(num_params=0)"
    assert embedder.num_params() == 0
    assert embedder.get_output_shape() == list(x.shape)
    assert embedder.get_output_dim(-1) == 6

    y = embedder.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == list(x.shape)
    printer(y.max(), y.min())
    assert y.max().item() <= 1
    assert y.min().item() >= 0

    assert torch.all(y == x)

def test_identity_model():
    printer = util.Printer("test_identity_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_identity_model")

    embedder = juml.models.embed.Identity()

    x = torch.rand([3, 4, 5, 6])
    t = torch.rand([3, 4, 5, 6])

    model = juml.models.Mlp(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        hidden_dim=13,
        depth=2,
        embedder=embedder,
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())
    assert model.embed is embedder

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
