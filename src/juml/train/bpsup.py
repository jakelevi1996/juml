import torch
from jutility import cli, util
from juml.train.base import Trainer
from juml.models.base import Model
from juml.datasets.base import Dataset

class BpSup(Trainer):
    def __init__(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
        optimiser:  torch.optim.Optimizer,
        batch_size: int,
        epochs:     int,
    ):

    @classmethod
    def init_sub_objects(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
    ):
        optimiser = args.init_object(
            "train.trainer.BpSup.optimiser",
            params=model.parameters(),
        )
        assert isinstance(optimiser, torch.optim.Optimizer)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("batch_size",   type=int, default=100),
            cli.Arg("epochs",       type=int, default=10),
            cli.ObjectChoice(
                "optimiser",
                cli.ObjectArg(
                    torch.optim.Adam,
                    cli.Arg("lr", type=float, default=0.001),
                ),
                cli.ObjectArg(
                    torch.optim.AdamW,
                    cli.Arg("lr",           type=float, default=0.001),
                    cli.Arg("weight_decay", type=float, default=0.01),
                ),
                default="Adam",
            ),
        ]
