import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_train/test_bpspde")

def test_bpspde():
    printer = util.Printer("test_bpspde", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_bpspde")

    parser  = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--trainer BpSpDe "
        "--trainer.BpSpDe.steps 10 "
        "--trainer.BpSpDe.n_train 150 "
        "--trainer.BpSpDe.batch_size 17 "
        "--model Mlp "
        "--model.Mlp.hidden_dim 11 "
        "--model.Mlp.depth 1 "
        "--model.Mlp.embedder Flatten "
        "--model.Mlp.embedder.Flatten.n 3 "
        "--dataset RandomClassification "
        "--dataset.RandomClassification.train 200 "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Train)

    trainer = command.run(args)
    printer(trainer.model)
    printer(trainer.table)
    assert isinstance(trainer,          juml.train.BpSpDe)
    assert isinstance(trainer.model,    juml.models.Mlp)
    assert isinstance(trainer.dataset,  juml.datasets.RandomClassification)
    assert isinstance(trainer.loss,     juml.loss.CrossEntropy)
    assert isinstance(trainer.table,    util.Table)

    batch_loss = trainer.table.get_data("batch_loss")
    printer(batch_loss)
    assert batch_loss[-1] < batch_loss[0]

    x, t = next(iter(trainer.dataset.get_data_loader("train", 13)))
    y = trainer.model.forward(x)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    assert list(x.shape) == [13, 3, 32, 32]
    assert list(y.shape) == [13, 10]
    assert list(t.shape) == [13]
