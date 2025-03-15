import torch
from jutility import util, cli
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_rzmlp")

def test_rzmlp():
    juml.test_utils.set_torch_seed("test_rzmlp")
    printer = util.Printer("test_rzmlp", dir_name=OUTPUT_DIR)

    input_dim   = 7
    output_dim  = 11
    n_train     = 50

    model_dim   = 23
    ratio       = 2
    hidden_dim  = model_dim * ratio

    dataset = juml.datasets.SinMix(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=3,
        train=n_train,
        test=60,
        x_std=0,
        t_std=0.1,
    )
    model = juml.models.RzMlp(
        input_shape=dataset.get_input_shape(),
        output_shape=dataset.get_output_shape(),
        model_dim=model_dim,
        expand_ratio=ratio,
        num_hidden_layers=3,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())

    x, t = next(iter(dataset.get_data_loader("train", 100)))
    assert isinstance(x, torch.Tensor)
    assert list(x.shape) == [n_train, input_dim]

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert list(y_0.shape) == [n_train, output_dim]

    loss_0 = loss.forward(y_0, t)
    loss_0.backward()
    optimiser.step()

    y_1 = model.forward(x)
    loss_1 = loss.forward(y_1, t)

    assert loss_1.item() < loss_0.item()

    for i in [1, 2]:
        layer = model.layers[i]
        assert isinstance(layer, juml.models.ReZeroMlpLayer)
        assert isinstance(layer.f1.w_io, torch.Tensor)
        assert list(layer.f1.w_io.shape) == [model_dim, hidden_dim]

        g = layer.f1.w_io.grad
        assert isinstance(g, torch.Tensor)
        assert list(g.shape) == [model_dim, hidden_dim]
