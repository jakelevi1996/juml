from jutility import cli, util
from juml import models, datasets, train, commands
from juml.models.base import Model
from juml.datasets.base import Dataset
from juml.train.base import Trainer
from juml.commands.base import Command

class Framework:
    @classmethod
    def get_models(cls) -> list[type[Model]]:
        return [
            models.Linear,
            models.Mlp,
            models.Cnn,
        ]

    @classmethod
    def get_datasets(cls) -> list[type[Dataset]]:
        return [
            datasets.Linear,
            datasets.Mnist,
            datasets.Cifar10,
        ]

    @classmethod
    def get_trainers(cls) -> list[type[Trainer]]:
        return [
            train.BpSp,
        ]

    @classmethod
    def get_defaults(cls) -> dict[str, str | None]:
        return {
            "model":    None,
            "dataset":  None,
            "trainer":  "BpSp",
        }

    @classmethod
    def get_train_args(cls) -> list[cli.Arg]:
        defaults = cls.get_defaults()
        return [
            cli.ObjectChoice(
                "model",
                *[t.get_cli_arg() for t in cls.get_models()],
                default=defaults["model"],
                is_group=True,
            ),
            cli.ObjectChoice(
                "dataset",
                *[t.get_cli_arg() for t in cls.get_datasets()],
                default=defaults["dataset"],
                is_group=True,
            ),
            cli.ObjectChoice(
                "trainer",
                *[t.get_cli_arg() for t in cls.get_trainers()],
                default=defaults["trainer"],
                is_group=True,
            ),
            cli.Arg("seed",         type=int, default=0),
            cli.Arg("gpu",          action="store_true"),
            cli.Arg("devices",      type=int, default=[],   nargs="*"),
            cli.Arg("configs",      type=str, default=[],   nargs="*"),
            cli.Arg("model_name",   type=str, default=None, is_kwarg=False),
        ]

    @classmethod
    def get_commands(cls) -> list[type[Command]]:
        return [
            commands.Train,
            commands.PlotConfusionMatrix,
        ]

    @classmethod
    def get_parser(cls) -> cli.Parser:
        return cli.Parser(
            sub_commands=cli.SubCommandGroup(
                *[
                    command_type.init_juml(cls.get_train_args())
                    for command_type in cls.get_commands()
                ],
            ),
        )

    @classmethod
    def run(cls, *parser_args, **parser_kwargs):
        parser = cls.get_parser()
        args = parser.parse_args(*parser_args, **parser_kwargs)
        command = args.get_command()

        with util.Timer(repr(command)):
            command.run(args)
