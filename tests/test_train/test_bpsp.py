import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_train/test_bpsp")

def test_bpsp():
    printer = util.Printer("test_bpsp", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_bpsp")

    parser  = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--trainer BpSp "
        "--trainer.BpSp.epochs 2 "
        "--trainer.BpSp.batch_size 57 "
        "--model Mlp "
        "--model.Mlp.hidden_dim 11 "
        "--model.Mlp.num_hidden_layers 1 "
        "--dataset SinMix "
        "--dataset.SinMix.input_dim 3 "
        "--dataset.SinMix.output_dim 5 "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Train)

    trainer = command.run(args)
    assert isinstance(trainer,          juml.train.BpSp)
    assert isinstance(trainer.model,    juml.models.Mlp)
    assert isinstance(trainer.dataset,  juml.datasets.SinMix)
    assert isinstance(trainer.loss,     juml.loss.Mse)
    assert isinstance(trainer.table,    util.Table)

    batch_loss = trainer.table.get_data("batch_loss")
    printer(batch_loss)
    assert batch_loss[-1] < batch_loss[0]

    x, t = next(iter(trainer.dataset.get_data_loader("train", 13)))
    y = trainer.model.forward(x)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    assert list(x.shape) == [13, 3]
    assert list(y.shape) == [13, 5]
    assert list(t.shape) == [13, 5]
