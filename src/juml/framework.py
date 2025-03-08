from jutility import cli, util
from juml import models, datasets, commands
from juml.train import modes
from juml.models.base import Model
from juml.datasets.base import Dataset
from juml.train.modes.base import Trainer
from juml.train.args import TrainArgs
from juml.commands.base import Command

class Framework:
    @classmethod
    def get_models(cls) -> list[type[Model]]:
        return [
            models.LinearModel,
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
            modes.BpSup,
            modes.BpSupDataEfficiency,
        ]

    @classmethod
    def get_defaults(cls) -> dict[str, str | None]:
        return {
            "model":    None,
            "dataset":  None,
            "trainer":  "BpSup",
        }

    @classmethod
    def get_train_arg(cls) -> cli.ObjectArg:
        return TrainArgs.get_cli_arg(
            cls.get_models(),
            cls.get_datasets(),
            cls.get_trainers(),
            cls.get_defaults(),
        )

    @classmethod
    def get_commands(cls) -> list[type[Command]]:
        return [
            commands.Train,
            commands.Sweep,
        ]

    @classmethod
    def run(cls, *parser_args, **parser_kwargs):
        parser = cli.Parser(
            sub_commands=cli.SubCommandGroup(
                *[
                    command_type.init_juml(cls.get_train_arg())
                    for command_type in cls.get_commands()
                ],
            ),
        )
        args = parser.parse_args(*parser_args, **parser_kwargs)
        command = args.get_command()

        with util.Timer(repr(command)):
            command.run(args)
