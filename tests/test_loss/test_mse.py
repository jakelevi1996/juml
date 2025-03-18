import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss/test_mse")

def test_mse():
    printer = util.Printer("test_mse", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mse")

    batch_size = 87
    output_dim = 11
    y = torch.rand([batch_size, output_dim])
    t = torch.rand([batch_size, output_dim])

    loss = juml.loss.Mse()

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
    assert isinstance(metric, float)
    assert metric >= 0
    assert metric <= 200

def test_mse_cli():
    printer = util.Printer("test_mse_cli", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mse_cli")

    parser  = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--dataset LinearDataset "
        "--model RzMlp "
        "--model.RzMlp.model_dim 7 "
        "--model.RzMlp.num_hidden_layers 2 "
        "--trainer BpSp "
        "--trainer.BpSp.epochs 1"
    )
    args = parser.parse_args(args_str.split())
    trainer = args.get_command().run(args)
    assert isinstance(trainer, juml.train.BpSp)

    loss = trainer.loss
    assert isinstance(loss, juml.loss.Mse)
    assert loss.weights is None
    assert not isinstance(loss.weights, torch.Tensor)

    batch_size = 11
    optimiser = torch.optim.Adam(trainer.model.parameters())
    x, t = next(iter(trainer.dataset.get_data_loader("train", batch_size)))
    y_0 = trainer.model.forward(x)
    loss_0 = loss.forward(y_0, t)

    optimiser.zero_grad()
    loss_0.backward()
    optimiser.step()

    y_1 = trainer.model.forward(x)
    loss_1 = loss.forward(y_1, t)

    printer(loss_0, loss_1)
    assert loss_1.item() < loss_0.item()
