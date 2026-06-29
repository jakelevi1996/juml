import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_cnn")

def test_cnn():
    printer = util.Printer("test_cnn", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_cnn")

    x = torch.rand([3, 4, 17, 19])
    t = torch.rand([3, 9, 3, 4])

    model = juml.models.Cnn(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        kernel_size=3,
        channel_dim=9,
        num_stages=2,
        blocks_per_stage=3,
        stride=2,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())

    assert repr(model) == "Cnn(num_params=4.8k)"
    assert model.num_params() == 4761

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
        "[InputReluCnnLayer(num_params=333), "
        "ReluCnnLayer(num_params=738), "
        "ReluCnnLayer(num_params=738), "
        "StridedReluCnnLayer(num_params=738), "
        "ReluCnnLayer(num_params=738), "
        "ReluCnnLayer(num_params=738), "
        "ReluCnnLayer(num_params=738)]"
    )
