import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_mlp")

def test_mlp():
    printer = util.Printer("test_mlp", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mlp")

    x = torch.rand([3, 4, 5, 7])
    t = torch.rand([3, 4, 5, 11])

    model = juml.models.Mlp(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        hidden_dim=13,
        depth=2,
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

    printer(repr(list(model.layers)))
    assert repr(list(model.layers)) == (
        "[ReluMlpLayer(num_params=104), "
        "ReluMlpLayer(num_params=182), "
        "Linear(num_params=154)]"
    )
