import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_linearmodel")

def test_linearmodel():
    printer = util.Printer("test_linearmodel", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_linearmodel")

    input_dim  = 7
    output_dim = 11
    x = torch.rand([3, 4, 5, input_dim])
    t = torch.rand([3, 4, 5, output_dim])

    model = juml.models.LinearModel(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())

    assert repr(model) == "LinearModel(num_params=88)"
    assert model.num_params() == (input_dim * output_dim) + output_dim

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
