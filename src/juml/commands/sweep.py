from jutility import cli
from juml.commands.base import Command
from juml.train.sweeper import Sweeper

class Sweep(Command):
    def run(self, args: cli.ParsedArgs):
        with cli.verbose:
            sweeper = args.init_object(
                "Sweeper",
                args=args,
                **self.get_kwargs(),
            )
            assert isinstance(sweeper, Sweeper)

        return sweeper

    @classmethod
    def get_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return [
            *[arg for arg in train_args if arg.name != "devices"],
            Sweeper.get_cli_arg(),
        ]
