import pytest
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss/test_chamfermse")

def test_chamfermse():
    printer = util.Printer("test_chamfermse", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_chamfermse")

    batch_size = 87
    y_set_size = 17
    t_set_size = 5
    output_dim = 11
    y = torch.rand([batch_size, y_set_size, output_dim])
    t = torch.rand([batch_size, t_set_size, output_dim])

    loss = juml.loss.ChamferMse()

    with pytest.raises(TypeError):
        batch_loss = loss.forward(y, t)

    loss.set_weights(t.flatten(0, -2).mean(dim=-2))
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

def test_chamfermse_cli():
    printer = util.Printer("test_chamfermse_cli", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_chamfermse_cli")

    parser  = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--loss ChamferMse "
        "--dataset RandomRegression "
        "--dataset.RandomRegression.output_shape 19 13 "
        "--model RzCnn "
        "--model.RzCnn.model_dim 7 "
        "--model.RzCnn.num_stages 2 "
        "--model.RzCnn.pooler LinearSet2d "
        "--trainer BpSp "
        "--trainer.BpSp.epochs 1"
    )
    args = parser.parse_args(args_str.split())
    trainer = args.get_command().run(args)
    assert isinstance(trainer, juml.train.BpSp)

    loss = trainer.loss
    assert isinstance(loss, juml.loss.ChamferMse)
    assert isinstance(loss.weights, torch.Tensor)
    assert loss.weights is not None

    batch_size = 11
    optimiser = torch.optim.Adam(trainer.model.parameters())
    x, t = next(iter(trainer.dataset.get_data_loader("train", batch_size)))
    y_0 = trainer.model.forward(x)
    loss_0 = loss.forward(y_0, t)
    assert isinstance(x,        torch.Tensor)
    assert isinstance(t,        torch.Tensor)
    assert isinstance(y_0,      torch.Tensor)
    assert isinstance(loss_0,   torch.Tensor)
    assert list(x.shape         ) == [11, 3, 32, 32]
    assert list(t.shape         ) == [11, 19, 13]
    assert list(y_0.shape       ) == [11, 25, 13]
    assert list(loss_0.shape    ) == []

    optimiser.zero_grad()
    loss_0.backward()
    optimiser.step()

    y_1 = trainer.model.forward(x)
    loss_1 = loss.forward(y_1, t)

    printer(loss_0, loss_1)
    assert loss_1.item() < loss_0.item()
