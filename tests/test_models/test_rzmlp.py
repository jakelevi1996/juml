import torch
from jutility import util, cli
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_rzmlp")

def test_rzmlp():
    juml.test_utils.set_torch_seed("test_rzmlp")
    printer = util.Printer("test_rzmlp", dir_name=OUTPUT_DIR)

    x = torch.rand([3, 4, 5, 7])
    t = torch.rand([3, 4, 5, 11])

    model_dim   = 23
    ratio       = 2
    hidden_dim  = model_dim * ratio

    model = juml.models.RzMlp(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        model_dim=model_dim,
        expand_ratio=ratio,
        num_hidden_layers=3,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())

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

    for i in [1, 2]:
        layer = model.layers[i]
        assert isinstance(layer, juml.models.ReZeroMlpLayer)
        assert isinstance(layer.f1.w_io, torch.Tensor)
        assert list(layer.f1.w_io.shape) == [model_dim, hidden_dim]

        g = layer.f1.w_io.grad
        assert isinstance(g, torch.Tensor)
        assert list(g.shape) == [model_dim, hidden_dim]

def test_rzmlp_num_params():
    juml.test_utils.set_torch_seed("test_rzmlp_num_params")
    printer = util.Printer("test_rzmlp_num_params", dir_name=OUTPUT_DIR)

    input_dim   = 7
    output_dim  = 11
    nhl         = 3
    model_dim   = 23
    ratio       = 2
    hidden_dim  = model_dim * ratio

    model = juml.models.RzMlp(
        input_shape=[input_dim],
        output_shape=[output_dim],
        model_dim=model_dim,
        expand_ratio=ratio,
        num_hidden_layers=nhl,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )

    def linear_layer_params(i, o):
        return i*o + o

    def rezero_mlp_layer_params(m, e):
        return linear_layer_params(m, e*m) + linear_layer_params(e*m, m)

    assert isinstance(model.num_params(), int)
    assert model.num_params() == (
        linear_layer_params(input_dim, model_dim)
        + (nhl * rezero_mlp_layer_params(model_dim, ratio))
        + linear_layer_params(model_dim, output_dim)
        + nhl
    )
    assert model.num_params() == 7006
    assert repr(model) == "RzMlp(num_params=7.0k)"
