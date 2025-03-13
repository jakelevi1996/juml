import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss")

def test_loss_metric():
    printer = util.Printer("test_loss_metric", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_loss_metric")

    input_dim   = 7
    output_dim  = 11
    n_train     = 23
    n_test      = 27
    batch_size  = 17

    model = juml.models.Mlp(
        input_shape=[input_dim],
        output_shape=[output_dim],
        hidden_dim=13,
        num_hidden_layers=2,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    dataset = juml.datasets.LinearDataset(
        input_dim=input_dim,
        output_dim=output_dim,
        n_train=n_train,
        n_test=n_test,
        x_std=0.1,
        t_std=0.2,
    )
    data_loader = dataset.get_data_loader("train", batch_size)

    metric = dataset.loss.metric(
        model=model,
        data_loader=data_loader,
        gpu=False,
    )
    assert isinstance(metric, float)
    assert metric > 0

def test_crossentropy():
    printer = util.Printer("test_crossentropy", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_crossentropy")

    output_dim = 11
    batch_size = 87

    y = torch.normal(0, 1, [batch_size, output_dim])
    t = torch.randint(0, output_dim, [batch_size])

    ce_loss = juml.datasets.loss.CrossEntropy()

    loss = ce_loss.forward(y, t)
    assert isinstance(loss, torch.Tensor)
    assert list(loss.shape) == []
    assert loss.item() >= 0

    metric = ce_loss.metric_batch(y, t)
    assert isinstance(metric, int)
    assert metric >= 0
    assert metric < batch_size

def test_mse():
    printer = util.Printer("test_mse", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_mse")

    output_dim = 11
    batch_size = 87

    y = torch.normal(0, 1, [batch_size, output_dim])
    t = torch.normal(0, 1, [batch_size, output_dim])

    mse_loss = juml.datasets.loss.Mse()

    loss = mse_loss.forward(y, t)
    assert isinstance(loss, torch.Tensor)
    assert list(loss.shape) == []
    assert loss.item() >= 0

    metric = mse_loss.metric_batch(y, t)
    assert isinstance(metric, float)
    assert metric >= 0
