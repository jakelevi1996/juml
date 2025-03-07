import math
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_cnn")

def test_cnn_identity_avg():
    printer = util.Printer("test_cnn_identity_avg", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_cnn_identity_avg")

    x = torch.rand([3, 4, 5, 6])
    t = torch.rand([3, 11])

    model = juml.models.Cnn(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        kernel_size=3,
        channel_dim=9,
        blocks=[3, 3],
        stride=2,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Average2d(),
    )
    assert repr(model) == "Cnn(num_params=4.9k)"
    assert model.num_params() == 4871

    y = model.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, 11]

    assert repr(model.embed) == "Identity(num_params=0)"
    assert repr(list(model.layers)) == (
        "[InputReluCnnLayer(num_params=333), ReluCnnLayer(num_params=738), "
        "ReluCnnLayer(num_params=738), StridedReluCnnLayer(num_params=738), "
        "ReluCnnLayer(num_params=738), ReluCnnLayer(num_params=738), "
        "ReluCnnLayer(num_params=738)]"
    )
    assert repr(model.pool) == "Average2d(num_params=110)"
