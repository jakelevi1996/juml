import torch
from jutility import util
import juml

def test_tensorprinter():
    x = torch.arange(20).reshape(4, 5)

    assert util.strings_equal_except_whitespace(
        juml.util.TensorPrinter.format(x),
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
        juml.util.TensorPrinter.format(x.float()),
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

def test_tensor_list_to_vector():
    juml.util.set_torch_seed("test_tensor_list_to_vector")

    model = juml.models.ReluMlp(3, 5, 10, 4)

    w = juml.util.tensor_list_to_vector(model.parameters())

    assert w.numel() == model.get_num_params()
    assert str(w.dtype) == "torch.float32"

def test_vector_to_tensor_list():
    juml.util.set_torch_seed("test_vector_to_tensor_list")

    model = juml.models.ReluMlp(3, 5, 10, 4)

    w = juml.util.tensor_list_to_vector(model.parameters())
    vector_info = juml.util.tensor_list_to_vector_info(model.parameters())

    x_list = juml.util.vector_to_tensor_list(w, vector_info)

    p_list = list(model.parameters())
    assert len(x_list) == len(p_list)

    for x, p in zip(x_list, p_list):
        assert list(x.shape) == list(p.shape)
        assert str(x.detach()) == str(p.detach())
