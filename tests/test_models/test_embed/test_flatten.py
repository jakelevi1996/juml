import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_embed")

def test_flatten():
    printer = util.Printer("test_flatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_flatten")

    embedder = juml.models.embed.Flatten(n=2)

    x = torch.rand([3, 4, 5, 6])

    embedder.set_input_shape(list(x.shape))
    assert repr(embedder) == "Flatten(num_params=0)"
    assert embedder.num_params() == 0
    assert embedder.get_output_shape() == [3, 4, 30]
    assert embedder.get_output_dim(-1) == 30

    y = embedder.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, 4, 30]
    printer(y.max(), y.min())
    assert y.max().item() <= 1
    assert y.min().item() >= 0

def test_flatten_model():
    printer = util.Printer("test_flatten_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_flatten_model")

    embedder = juml.models.embed.Flatten(n=2)

    x = torch.rand([3, 4, 5, 6])
    t = torch.rand([3, 4, 17])

    model = juml.models.Mlp(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        hidden_dim=13,
        num_hidden_layers=2,
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

def test_flatten_different_n():
    printer = util.Printer("test_flatten_different_n", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_flatten_different_n")

    x = torch.rand([3, 4, 5, 6])

    e1 = juml.models.embed.Flatten(n=1)
    e2 = juml.models.embed.Flatten(n=2)
    e3 = juml.models.embed.Flatten(n=3)

    e1.set_input_shape(list(x.shape))
    e2.set_input_shape(list(x.shape))
    e3.set_input_shape(list(x.shape))

    assert list(e1.forward(x).shape) == [3, 4, 5, 6]
    assert list(e2.forward(x).shape) == [3, 4, 30]
    assert list(e3.forward(x).shape) == [3, 120]

    assert torch.all(e1.forward(x) == x)
    assert torch.all(e2.forward(x).reshape(x.shape) == x)
    assert torch.all(e3.forward(x).reshape(x.shape) == x)

    assert e1.get_output_shape() == [3, 4, 5, 6]
    assert e2.get_output_shape() == [3, 4, 30]
    assert e3.get_output_shape() == [3, 120]

    assert e1.get_output_dim(-1) == 6
    assert e2.get_output_dim(-1) == 30
    assert e3.get_output_dim(-1) == 120
