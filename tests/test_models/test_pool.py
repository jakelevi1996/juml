import math
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_pool")

def test_identity():
    printer = util.Printer("test_identity", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_identity")

    pooler = juml.models.pool.Identity()
    assert repr(pooler) == "Identity(num_params=0)"

    pooler.set_shapes([1, 2, 3], [4, 5, 6, 7])
    assert pooler.get_input_shape() == [4, 5, 6, 7]
    assert pooler.get_input_dim(-1) == 7

    x = torch.rand([8, 9])
    y = pooler.forward(x)
    assert list(y.shape) == list(x.shape)
    assert torch.all(y == x)

def test_unflatten():
    printer = util.Printer("test_unflatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_unflatten")

    pooler = juml.models.pool.Unflatten(3)
    assert repr(pooler) == "Unflatten(num_params=0)"

    pooler.set_shapes([], [8, 9, 2, 3, 4])
    assert pooler.get_input_shape() == [8, 9, (2 * 3 * 4)]
    assert pooler.get_input_dim(-1) == (2 * 3 * 4)

    x = torch.rand([1, (2 * 3 * 4)])
    y = pooler.forward(x)
    assert list(y.shape) == [1, 2, 3, 4]
    assert torch.all(y.flatten(-3, -1) == x)

def test_average2d():
    printer = util.Printer("test_average2d", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_average2d")

    pooler = juml.models.pool.Average2d()
    assert repr(pooler) == "Average2d(num_params=0)"

    pooler.set_shapes([3, 4, 5, 6], [7, 8])
    x = torch.rand([6, 4, 3, 2])
    y = pooler.forward(x)
    assert list(y.shape) == [6, 8]
