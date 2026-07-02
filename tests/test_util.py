import torch
from jutility import util
import juml

def test_tensorprinter():
    x = torch.arange(20).reshape(4, 5)

    assert util.strings_equal_except_whitespace(
        juml.util.TensorPrinter.format(x, None, " "),
        (
            "shape = [4, 5]"
            "numel = 20"
            "dtype = torch.int64"
            "tensor([[ 0,  1,  2,  3,  4],"
            "        [ 5,  6,  7,  8,  9],"
            "        [10, 11, 12, 13, 14],"
            "        [15, 16, 17, 18, 19]])"
        ),
    )
    assert util.strings_equal_except_whitespace(
        juml.util.TensorPrinter.format(x.float(), None, " "),
        (
            "shape = [4, 5]"
            "numel = 20"
            "dtype = torch.float32"
            "tensor([[ 0.,  1.,  2.,  3.,  4.],"
            "        [ 5.,  6.,  7.,  8.,  9.],"
            "        [10., 11., 12., 13., 14.],"
            "        [15., 16., 17., 18., 19.]])"
        ),
    )
