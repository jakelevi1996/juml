import torch
from jutility import plotting, util, cli
from juml.commands.command import Command
from juml.data import ClassificationDataset, get_all_datasets
from juml.models import ClassificationModel, get_all_models
from juml.util import softmax_cross_entropy_from_logits, multiclass_acc

class TrainClassification(Command):
    def run(
        self,
        seed:       int,
        epochs:     int,
        batch_size: int,
    ):
        torch.manual_seed(seed)

        dataset = self.init_object("dataset")
        assert isinstance(dataset, ClassificationDataset)

        model = self.init_object(
            "model",
            input_dim=dataset.get_input_dim(),
            output_dim=dataset.get_output_dim(),
        )
        assert isinstance(model, ClassificationModel)

        opt = torch.optim.Adam(model.parameters())

        train_loader = dataset.get_data_loader("train", batch_size)
        x_train, t_train = dataset.get_full_batch("train")
        x_test,  t_test  = dataset.get_full_batch("test")
        x_train, t_train = dataset.format_batch(x_train, t_train)
        x_test,  t_test  = dataset.format_batch(x_test,  t_test)

        table = util.Table(
            util.CountColumn(),
            util.TimeColumn(),
            util.Column("epoch",        "i",    width=10),
            util.Column("batch",        "i",    width=10),
            util.Column("loss",         ".5f",  width=10),
            util.Column("train_acc",    ".5f",  width=10),
            util.Column("test_acc",     ".5f",  width=10),
            print_interval=util.TimeInterval(1),
        )

        for e in range(epochs):
            for b, (x, t) in enumerate(train_loader):
                x, t = dataset.format_batch(x, t)
                y = model.forward(x)

                loss = softmax_cross_entropy_from_logits(y, t, -1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

                table.update(
                    epoch=e,
                    batch=b,
                    loss=loss.item(),
                )

            table.print_last()

            y_train     = model.forward(x_train)
            y_test      = model.forward(x_test)
            train_acc   = multiclass_acc(y_train, t_train).item()
            test_acc    = multiclass_acc(y_test,  t_test).item()
            table.update(
                epoch=e,
                train_acc=train_acc,
                test_acc=test_acc,
                level=1,
            )

        self.save_args()
        self.save_cmd()

        train_loss  = table.get_data("loss")
        train_acc   = table.get_data("train_acc")
        test_acc    = table.get_data("test_acc")
        t           = table.get_item("t", -1)
        self.save_metrics(
            final_train_loss    =train_loss[-1],
            final_train_acc     =train_acc[-1],
            final_test_acc      =test_acc[-1],
            min_train_loss      =min(train_loss),
            min_train_acc       =min(train_acc),
            min_test_acc        =min(test_acc),
            max_train_loss      =max(train_loss),
            max_train_acc       =max(train_acc),
            max_test_acc        =max(test_acc),
            time                =t,
            time_str            =util.time_format(t, True),
        )

        output_dir = self.get_output_dir()

        mp = plotting.MultiPlot(
            plotting.Subplot(
                plotting.Line(table.get_data("loss")),
                log_y=True,
                title="Loss",
            ),
            plotting.Subplot(
                plotting.Line(table.get_data("train_acc"),  c="b", m="o"),
                plotting.Line(table.get_data("test_acc"),   c="r", m="o"),
                ylim=[None, 1],
                title="Accuracy",
            ),
            legend=plotting.FigureLegend.centre_right(
                plotting.Line(c="b", m="o", label="Train"),
                plotting.Line(c="r", m="o", label="Test"),
            ),
            figsize=[8, 3],
        )
        mp.save("metrics", output_dir)

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return [
            cli.Arg("seed",         type=int,   default=0),
            cli.Arg("epochs",       type=int,   default=1),
            cli.Arg("batch_size",   type=int,   default=100),
            cli.ObjectChoice(
                "dataset",
                *[
                    dataset_type.get_cli_arg()
                    for dataset_type in get_all_datasets()
                ],
                is_group=True,
            ),
            cli.ObjectChoice(
                "model",
                *[
                    model_type.get_cli_arg()
                    for model_type in get_all_models()
                ],
                is_group=True,
            ),
        ]
