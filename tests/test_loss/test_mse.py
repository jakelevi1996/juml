import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss/test_mse")

def test_mse():
    printer = util.Printer("test_mse", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mse")

    output_dim = 11
    batch_size = 87

    y = torch.normal(0, 1, [batch_size, output_dim])
    t = torch.normal(0, 1, [batch_size, output_dim])

    mse_loss = juml.loss.Mse()

    loss = mse_loss.forward(y, t)
    assert isinstance(loss, torch.Tensor)
    assert list(loss.shape) == []
    assert loss.item() >= 0

    metric = mse_loss.metric_batch(y, t)
    assert isinstance(metric, float)
    assert metric >= 0
