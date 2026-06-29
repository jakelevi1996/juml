import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_tools/test_display")

def test_display_sequential():
    printer = util.Printer("test_display_sequential", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_display_sequential")

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
        embedder=juml.models.embed.CoordConv(),
        pooler=juml.models.pool.Identity(),
    )

    y, table = juml.tools.display_sequential(model, x, printer)

    output_line = (
        "ReZeroCnnLayer(num_params=442)                  "
        "| [3, 9, 8, 9]           |    0.0"
    )
    assert output_line in printer.read()

    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == list(t.shape)
    printer(y.max(), y.min())
    assert y.max().item() <= 2
    assert y.min().item() >= -2

    assert isinstance(table, util.Table)
