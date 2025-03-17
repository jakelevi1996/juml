import math
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_embed")

def test_identity():
    printer = util.Printer("test_identity", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_identity")

    input_shape = [3, 4, 5, 6]
    x = torch.rand(input_shape)

    e = juml.models.embed.Identity()
    assert list(x.shape) == input_shape
    assert list(e.forward(x).shape) == input_shape
    assert torch.all(e.forward(x) == x)

    e.set_input_shape(input_shape)
    assert list(e.get_output_shape()) == input_shape

def test_flatten():
    printer = util.Printer("test_flatten", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_flatten")

    input_shape = [3, 4, 5, 6]
    x = torch.rand(input_shape)

    e1 = juml.models.embed.Flatten(1)
    e2 = juml.models.embed.Flatten(2)
    e3 = juml.models.embed.Flatten(3)

    assert list(x.shape) == input_shape
    assert list(e1.forward(x).shape) == [3, 4, 5, 6]
    assert list(e2.forward(x).shape) == [3, 4, 30]
    assert list(e3.forward(x).shape) == [3, 120]

    assert torch.all(e1.forward(x) == x)
    assert torch.all(e2.forward(x).reshape(x.shape) == x)
    assert torch.all(e3.forward(x).reshape(x.shape) == x)

    e1.set_input_shape(input_shape)
    e2.set_input_shape(input_shape)
    e3.set_input_shape(input_shape)
    assert list(e1.get_output_shape()) == [3, 4, 5, 6]
    assert list(e2.get_output_shape()) == [3, 4, 30]
    assert list(e3.get_output_shape()) == [3, 120]
