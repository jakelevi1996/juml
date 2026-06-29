import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss/test_base")

def test_loss_metric():
    printer = util.Printer("test_loss_metric", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_loss_metric")

    loss = juml.loss.Mse()

    input_dim   = 7
    output_dim  = 11
    n_train     = 23
    n_test      = 27
    batch_size  = 17

    model = juml.models.Mlp(
        input_shape=[input_dim],
        output_shape=[output_dim],
        hidden_dim=13,
        depth=2,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    dataset = juml.datasets.LinearDataset(
        input_dim=input_dim,
        output_dim=output_dim,
        train=n_train,
        test=n_test,
        x_std=0.1,
        t_std=0.2,
    )
    data_loader = dataset.get_data_loader("train", batch_size)

    metric = loss.metric(
        model=model,
        data_loader=data_loader,
        device_cfg=juml.device.DeviceConfig([]),
    )
    assert isinstance(metric, float)
    assert metric > 0
