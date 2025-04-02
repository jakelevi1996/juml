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
    def get_ignore_args(cls) -> set[str]:
        return {"devices"}

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return Sweeper.get_cli_options()
