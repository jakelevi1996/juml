import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss/test_crossentropy")

def test_crossentropy():
    printer = util.Printer("test_crossentropy", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_crossentropy")

    batch_size  = 87
    num_classes = 11
    y = torch.rand([batch_size, num_classes])
    t = torch.randint(0, num_classes, [batch_size])

    loss = juml.loss.CrossEntropy()

    batch_loss = loss.forward(y, t)
    assert isinstance(batch_loss, torch.Tensor)
    assert batch_loss.dtype is torch.float32
    assert batch_loss.dtype is not torch.int64
    assert list(batch_loss.shape) == []
    printer(batch_loss)
    assert batch_loss.item() >= 0
    assert batch_loss.item() <= 3

    metric = loss.metric_batch(y, t)
    printer(metric)
    assert isinstance(metric, int)
    assert metric >= 0
    assert metric <= batch_size
