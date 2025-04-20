import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_embed")

def test_coordconv():
    printer = util.Printer("test_coordconv", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_coordconv")

    embedder = juml.models.embed.CoordConv()

    x = torch.rand([2, 4, 3, 5])

    embedder.set_input_shape(list(x.shape))
    assert repr(embedder) == "CoordConv(num_params=0)"
    assert sum(int(p.numel()) for p in embedder.parameters()) == (3 * 5 * 2)
    assert embedder.num_params() == 0
    assert embedder.get_output_shape() == [2, 6, 3, 5]
    assert embedder.get_output_dim(-3) == 6

    y = embedder.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [2, 6, 3, 5]
    printer(y.max(), y.min())
    assert y.max().item() == 1
    assert y.min().item() == -1

    yw = torch.linspace(-1, 1, x.shape[-1]).unsqueeze(-2)
    yh = torch.linspace(-1, 1, x.shape[-2]).unsqueeze(-1)
    assert torch.all(y[:, :-2, :, :] == x)
    assert torch.all(y[:,  -2, :, :] == yw)
    assert torch.all(y[:,  -1, :, :] == yh)

def test_coordconv_model():
    printer = util.Printer("test_coordconv_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_coordconv_model")

    embedder = juml.models.embed.CoordConv()

    x = torch.rand([3, 4, 17, 19])
    t = torch.rand([3, 9, 3, 4])

    model = juml.models.RzCnn(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        kernel_size=3,
        model_dim=9,
        expand_ratio=2.0,
        num_stages=2,
        blocks_per_stage=3,
        stride=2,
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
