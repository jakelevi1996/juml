from jutility import cli
from juml.commands.base import Command
from juml.tools.sweeper import Sweeper

class Sweep(Command):
    def run(
        self,
        args: cli.ParsedArgs,
        **kwargs,
    ):
        return Sweeper(
            args=args,
            **kwargs,
            **args.get_kwargs(),
        )

    @classmethod
    def select_train_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return [
            arg
            for arg in train_args
            if  arg.name != "devices"
        ]

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return Sweeper.get_cli_options()
