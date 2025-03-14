import torch
from jutility import cli, util
from juml import device
from juml.train.base import Trainer
from juml.models.base import Model
from juml.datasets.base import Dataset

class BpSp(Trainer):
    def __init__(
        self,
        args:           cli.ParsedArgs,
        model:          Model,
        dataset:        Dataset,
        gpu:            bool,
        table:          util.Table,
        batch_size:     int,
        epochs:         int,
        optimiser:      torch.optim.Optimizer,
        lrs:            torch.optim.lr_scheduler.LRScheduler,
    ):
        self._init_trainer(model, dataset, table)

        train_loader = dataset.get_data_loader("train", batch_size)
        test_loader  = dataset.get_data_loader("test" , batch_size)

        table.get_column("train_metric").set_callback(
            lambda: dataset.loss.metric(model, train_loader, gpu),
            level=1,
        )
        table.get_column("test_metric").set_callback(
            lambda: dataset.loss.metric(model, test_loader, gpu),
            level=1,
        )

        for e in range(epochs):
            table.update(level=1, epoch=e)
            for i, (x, t) in enumerate(train_loader):
                x, t = device.to_device([x, t], gpu)
                y = model.forward(x)
                loss = dataset.loss.forward(y, t)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                table.update(epoch=e, batch=i, batch_loss=loss.item())

            table.print_last()
            lrs.step()

        table.update(level=2, epoch=epochs)

        self.save_results(args, model, dataset, table)

    @classmethod
    def init_sub_objects(
        cls,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
    ):
        with cli.verbose:
            optimiser = args.init_object(
                "trainer.BpSp.optimiser",
                params=model.parameters(),
            )
            assert isinstance(optimiser, torch.optim.Optimizer)

        scheduler = args.init_object(
            "trainer.BpSp.lrs",
            optimizer=optimiser,
            T_max=args.get_value("trainer.BpSp.epochs"),
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)

    @classmethod
    def get_table_columns(cls) -> list[util.Column]:
        return [
            util.TimeColumn("t"),
            util.Column("epoch"),
            util.Column("batch"),
            util.Column("batch_loss", ".5f", width=10),
            util.CallbackColumn("train_metric", ".5f", width=12),
            util.CallbackColumn("test_metric",  ".5f", width=12),
        ]

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
                cli.ObjectArg(
                    torch.optim.SGD,
                    cli.Arg("lr",           type=float, default=0.001),
                    cli.Arg("momentum",     type=float, default=0.0),
                    cli.Arg("weight_decay", type=float, default=0.0),
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
