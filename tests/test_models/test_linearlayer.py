import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_linearlayer")

def test_init_batch_no_target():
    printer = util.Printer("test_init_batch_no_target", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_init_batch_no_target")
    juml.test_utils.torch_set_print_options()

    input_dim = 19
    output_dim = 7
    batch_size = 100

    def get_x():
        return torch.normal(1e3, 1e2, [batch_size, input_dim])

    x = get_x()
    m = juml.models.Linear(input_dim, output_dim)
    y = m.forward(x)
    printer(y.mean(dim=-2))
    printer(y.std(dim=-2))
    printer.hline()
    assert y.mean(dim=-2).abs().max()       > 1000
    assert (y.std(dim=-2) - 1).abs().max()  > 100

    m.init_batch(x, None)

    y = m.forward(x)
    printer(y.mean(dim=-2))
    printer(y.std(dim=-2))
    printer.hline()
    assert y.mean(dim=-2).abs().max()       < 1e-3
    assert (y.std(dim=-2) - 1).abs().max()  < 1e-3

    x = get_x()
    y = m.forward(x)
    printer(y.mean(dim=-2))
    printer(y.std(dim=-2))
    printer.hline()
    assert y.mean(dim=-2).abs().max()       < 0.4
    assert (y.std(dim=-2) - 1).abs().max()  < 0.4

def test_init_batch_with_target():
    printer = util.Printer("test_init_batch_with_target", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_init_batch_with_target")
    juml.test_utils.torch_set_print_options()

    input_dim = 19
    output_dim = 7
    batch_size = 100

    m = juml.models.Linear(input_dim, output_dim)

    loss = juml.loss.Mse()
    d = juml.datasets.LinearDataset(
        input_dim=input_dim,
        output_dim=output_dim,
        train=batch_size,
        test=batch_size,
        x_std=0,
        t_std=0,
    )
    x, t = next(iter(d.get_data_loader("train", batch_size)))
    y = m.forward(x)
    printer(loss.forward(y, t))
    assert loss.forward(y, t).item() > 100

    m.init_batch(x, t)

    y = m.forward(x)
    printer(loss.forward(y, t))
    printer(loss.forward(y, t).item())
    assert loss.forward(y, t).item() < 1e-5

    x, t = next(iter(d.get_data_loader("test", batch_size)))
    y = m.forward(x)
    printer(loss.forward(y, t))
    printer(loss.forward(y, t).item())
    assert loss.forward(y, t).item() < 1e-5
