import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_test_utils")

def test_all_in_range():
    juml.test_utils.set_torch_seed("test_all_in_range")

    x = torch.normal(0, 1, [5, 6])

    assert type(juml.test_utils.all_in_range(x, -3, 3)) is bool

    assert juml.test_utils.all_in_range(x, -3, 3)
    assert not juml.test_utils.all_in_range(x, 3, -3)
    assert not juml.test_utils.all_in_range(x, -1, 1)
    assert juml.test_utils.all_in_range(x, -10, 10)
    assert juml.test_utils.all_in_range(torch.zeros([2, 3, 4]), 0, 0)
    assert juml.test_utils.all_in_range(torch.zeros([2, 3, 4]), -1, 1)
    assert not juml.test_utils.all_in_range(torch.zeros([2, 3, 4]), 1, 1)
    assert not juml.test_utils.all_in_range(torch.zeros([2, 3, 4]), 1, -1)

    assert juml.test_utils.all_in_range(x, -5, 5)
    x[0, 0] = torch.nan
    assert not juml.test_utils.all_in_range(x, -5, 5)
    x[0, 0] = 0
    assert juml.test_utils.all_in_range(x, -5, 5)
    x[0, 0] /= 0
    assert not juml.test_utils.all_in_range(x, -5, 5)
    x[0, 0] = 0
    assert juml.test_utils.all_in_range(x, -5, 5)
    x[0, 0] = 20
    assert not juml.test_utils.all_in_range(x, -5, 5)

def test_all_close_to_zero():
    juml.test_utils.set_torch_seed("test_all_close_to_zero")

    assert juml.test_utils.all_close_to_zero(torch.zeros([2, 3, 4]))
    assert not juml.test_utils.all_close_to_zero(torch.ones( [2, 3, 4]))

    x = torch.normal(0, 1, [5, 10])

    assert not juml.test_utils.all_close_to_zero(x)
    assert juml.test_utils.all_close_to_zero(x, tol=10)
    assert juml.test_utils.all_close_to_zero(x * 1e-7)

    x[0, 0] = 1e-7
    assert not juml.test_utils.all_close_to_zero(x.abs())
    assert not juml.test_utils.all_close_to_zero(-x.abs())

def test_all_close():
    juml.test_utils.set_torch_seed("test_all_close")

    assert juml.test_utils.all_close(torch.zeros([2, 3, 4]), 0)
    assert juml.test_utils.all_close(torch.ones( [2, 3, 4]), 1)
    assert not juml.test_utils.all_close(torch.zeros([2, 3, 4]), 1)
    assert not juml.test_utils.all_close(torch.ones( [2, 3, 4]), 0)
    assert juml.test_utils.all_close(torch.zeros([2, 3, 4]), 1, tol=10)
    assert juml.test_utils.all_close(torch.ones( [2, 3, 4]), 0, tol=10)

    x = torch.normal(0, 1, [5, 10])
    y = torch.normal(0, 1, [5, 10])

    assert not juml.test_utils.all_close(x, y)
    assert not juml.test_utils.all_close(x, x + y)
    assert juml.test_utils.all_close(x, x)
    assert juml.test_utils.all_close(x, x + 1e-7 * y)
