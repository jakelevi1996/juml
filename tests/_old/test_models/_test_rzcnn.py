import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_rzcnn")

def test_rzcnn():
    printer = util.Printer("test_rzcnn", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_rzcnn")

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
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.Mse()
    optimiser = torch.optim.Adam(model.parameters())

    assert repr(model) == "RzCnn(num_params=3.3k)"
    assert model.num_params() == 3281

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

    layer_names = [repr(layer) for layer in model.layers]
    printer.hline()
    printer(*layer_names, sep="\n")
    assert layer_names == [
        "Conv2d(4, 9, kernel_size=(3, 3), stride=(2, 2))",
        "ReZeroCnnLayer(num_params=442)",
        "ReZeroCnnLayer(num_params=442)",
        "Conv2d(9, 9, kernel_size=(3, 3), stride=(2, 2))",
        "ReZeroCnnLayer(num_params=442)",
        "ReZeroCnnLayer(num_params=442)",
        "ReZeroCnnLayer(num_params=442)",
    ]

    layer_sizes = [
        sum(int(p.numel()) for p in layer.parameters())
        for layer in model.layers
    ]
    printer.hline()
    printer(layer_sizes)
    assert layer_sizes == [333, 442, 442, 738, 442, 442, 442]
