import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss/test_crossentropy")

def test_crossentropy():
    printer = util.Printer("test_crossentropy", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_crossentropy")

    output_dim = 11
    batch_size = 87

    y = torch.normal(0, 1, [batch_size, output_dim])
    t = torch.randint(0, output_dim, [batch_size])

    ce_loss = juml.loss.CrossEntropy()

    loss = ce_loss.forward(y, t)
    assert isinstance(loss, torch.Tensor)
    assert list(loss.shape) == []
    assert loss.item() >= 0

    metric = ce_loss.metric_batch(y, t)
    assert isinstance(metric, int)
    assert metric >= 0
    assert metric < batch_size
