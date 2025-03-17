import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_mlp")

def test_mlp():
    printer = util.Printer("test_mlp", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mlp")

    input_dim  = 7
    output_dim = 11
    x = torch.rand([3, 4, 5, input_dim])
    t = torch.rand([3, 4, 5, output_dim])

    model = juml.models.Mlp(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        hidden_dim=13,
        num_hidden_layers=2,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())

    assert repr(model) == "Mlp(num_params=440)"
    assert model.num_params() == 440

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert y_0.dtype is torch.float32
    assert y_0.dtype is not torch.int64
    assert list(y_0.shape) == [3, 4, 5, output_dim]
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
        embedder=juml.models.embed.Flatten(n=3),
        pooler=juml.models.pool.Identity(),
    )
    assert repr(model) == "Mlp(num_params=2.2k)"
    assert model.num_params() == 2169

    y = model.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, output_dim]

def test_mlp_unflatten():
    printer = util.Printer("test_mlp_unflatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mlp_unflatten")

    x = torch.rand([3, 4, 5, 6])
    t = torch.rand([3, 4, 5, 7])

    pooler = juml.models.pool.Unflatten(2)
    embedder = juml.models.embed.Flatten(2)
    model = juml.models.Mlp(
        input_shape=x.shape,
        output_shape=t.shape,
        hidden_dim=20,
        num_hidden_layers=2,
        embedder=embedder,
        pooler=pooler,
    )
    assert pooler.get_input_shape() == [3, 4, (5 * 7)]
    assert pooler.get_input_dim(-1) == (5 * 7)
    assert repr(embedder)   == "Flatten(num_params=0)"
    assert repr(pooler)     == "Unflatten(num_params=0)"
    assert repr(model)      == "Mlp(num_params=1.8k)"

    output_layer = model.layers[-1]
    assert isinstance(output_layer, juml.models.Linear)
    assert repr(output_layer) == "Linear(num_params=735)"
    assert list(output_layer.w_io.shape) == [20, 35]
    assert list(output_layer.b_o.shape)  == [35]

    y = model.forward(x)
    assert list(y.shape) == [3, 4, 5, 7]
