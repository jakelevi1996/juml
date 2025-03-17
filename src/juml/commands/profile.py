from jutility import cli
from juml.commands.base import Command
from juml.train.profiler import Profiler

class Profile(Command):
    def run(self, args: cli.ParsedArgs) -> Profiler:
        with cli.verbose:
            profiler = args.init_object(
                "Profiler",
                args=args,
            )
            assert isinstance(profiler, Profiler)

        return profiler

    @classmethod
    def get_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return [
            *train_args,
            Profiler.get_cli_arg(),
        ]
