import torch
from jutility import cli, util
from juml.train.base import Trainer
from juml.models.base import Model
from juml.datasets.base import Dataset

class BpSup(Trainer):
    def __init__(
        self,
        args:           cli.ParsedArgs,
        model:          Model,
        dataset:        Dataset,
        optimiser:      torch.optim.Optimizer,
        lrs:            torch.optim.lr_scheduler.LRScheduler,
        seed:           int,
        batch_size:     int,
        epochs:         int,
        gpu:            bool,
        devices:        list[int],
        configs:        list[str],
        print_level:    int,
    ):

    @classmethod
    def init_sub_objects(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
    ):
        optimiser = args.init_object(
            "trainer.BpSup.optimiser",
            params=model.parameters(),
        )
        assert isinstance(optimiser, torch.optim.Optimizer)

        scheduler = args.init_object(
            "trainer.BpSup.lrs",
            optimizer=optimiser,
            T_max=args.get_value("trainer.BpSup.epochs"),
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("seed",         type=int, default=0),
            cli.Arg("batch_size",   type=int, default=100),
            cli.Arg("epochs",       type=int, default=10),
            cli.NoTagArg("gpu",          action="store_true"),
            cli.NoTagArg("devices",      type=int, default=[], nargs="*"),
            cli.NoTagArg("configs",      type=str, default=[], nargs="*"),
            cli.NoTagArg("print_level",  type=int, default=0),
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
            cli.ObjectChoice(
                "lrs",
                cli.ObjectArg(
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                    cli.Arg("eta_min", type=float, default=1e-5),
                ),
                default="CosineAnnealingLR",
            ),
        ]
