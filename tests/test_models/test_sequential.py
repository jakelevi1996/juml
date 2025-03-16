import torch
import pytest
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_sequential")

def test_sequential():
    printer = util.Printer("test_sequential", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_sequential")

    x = torch.rand([3, 4, 5, 6])
    t = torch.rand([3, 10, 11])

    model = juml.models.Cnn(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        kernel_size=3,
        channel_dim=9,
        num_stages=2,
        blocks_per_stage=3,
        stride=2,
        embedder=juml.models.embed.CoordConv(),
        pooler=juml.models.pool.SigmoidProduct2d(),
    )
    assert repr(model) == "Cnn(num_params=5.1k)"
    assert len(model) == 7
    assert model.num_params() == 5103

    p, s = model.split(5)

    assert isinstance(p, juml.models.Sequential)
    assert isinstance(s, juml.models.Sequential)
    assert len(p) == 5
    assert len(s) == 2
    assert len(p) + len(s) == len(model)
    assert p.num_params() == 3507
    assert s.num_params() == 1596
    assert p.num_params() + s.num_params() == model.num_params()
    assert repr(p) == "Sequential(num_params=3.5k)"
    assert repr(s) == "Sequential(num_params=1.6k)"

    y  = model.forward(x)
    y1 = p.forward(x)
    y2 = s.forward(y1)

    assert isinstance(y1, torch.Tensor)
    assert y1.dtype is torch.float32
    assert y1.dtype is not torch.int64
    assert list(y1.shape) == [3, 9, 2, 2]

    assert isinstance(y2, torch.Tensor)
    assert y2.dtype is torch.float32
    assert y2.dtype is not torch.int64
    assert list(y2.shape) == list(y.shape)
    assert torch.all(y2 == y)

    with pytest.raises(RuntimeError):
        y1 = s.forward(x)

    with pytest.raises(RuntimeError):
        y2 = p.forward(y1)
