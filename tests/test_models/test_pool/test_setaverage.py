import pytest
import torch
from jutility import util
import juml
import juml.models.rzcnn

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_pool")

def test_setaverage():
    printer = util.Printer("test_setaverage", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_setaverage")

    pooler = juml.models.pool.SetAverage()

    x = torch.rand([3, 11, 6])

    pooler.set_shapes(list(x.shape), [11])
    assert repr(pooler) == "SetAverage(num_params=77)"
    assert pooler.num_params() == (6*11 + 11)
    assert pooler.num_params() == 77
    with pytest.raises(NotImplementedError):
        pooler.get_input_shape()
    with pytest.raises(NotImplementedError):
        pooler.get_input_dim(-1)

    y = pooler.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == [3, 11]
    printer(y.max(), y.min())
    assert y.max().item() <= 2
    assert y.min().item() >= -2
