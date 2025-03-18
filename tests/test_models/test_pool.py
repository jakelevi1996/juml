import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_pool")

def test_average2d():
    printer = util.Printer("test_average2d", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_average2d")

    pooler = juml.models.pool.Average2d()
    pooler.set_shapes([3, 4, 5, 6], [7, 8])
    assert repr(pooler) == "Average2d(num_params=40)"

    x = torch.rand([6, 4, 3, 2])
    y = pooler.forward(x)
    assert list(y.shape) == [6, 8]

def test_max2d():
    printer = util.Printer("test_max2d", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_max2d")

    pooler = juml.models.pool.Max2d()
    pooler.set_shapes([3, 4, 5, 6], [7, 8])
    assert repr(pooler) == "Max2d(num_params=40)"

    x = torch.rand([6, 4, 3, 2])
    y = pooler.forward(x)
    assert list(y.shape) == [6, 8]

def test_attention2d():
    printer = util.Printer("test_attention2d", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_attention2d")

    pooler = juml.models.pool.Attention2d()
    pooler.set_shapes([3, 4, 5, 6], [7, 8])
    assert repr(pooler) == "Attention2d(num_params=45)"

    x = torch.rand([6, 4, 3, 2])
    y = pooler.forward(x)
    assert list(y.shape) == [6, 8]

def test_conv2d():
    printer = util.Printer("test_conv2d", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_conv2d")

    pooler = juml.models.pool.Conv2d()
    pooler.set_shapes([3, 4, 5, 6], [7, 8])
    assert repr(pooler) == "Conv2d(num_params=40)"

    x = torch.rand([6, 4, 3, 2])
    y = pooler.forward(x)
    assert list(y.shape) == [6, 8, 3, 2]
