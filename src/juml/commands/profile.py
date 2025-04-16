from jutility import cli
from juml.commands.base import Command
from juml.train.profiler import Profiler

class Profile(Command):
    @classmethod
    def run(
        cls,
        args:           cli.ParsedArgs,
        batch_size:     int,
        num_warmup:     int,
        num_profile:    int,
        devices:        list[int],
    ):
        return Profiler(
            args=args,
            batch_size=batch_size,
            num_warmup=num_warmup,
            num_profile=num_profile,
            devices=devices,
        )

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("batch_size",   type=int, default=100),
            cli.Arg("num_warmup",   type=int, default=10),
            cli.Arg("num_profile",  type=int, default=10),
            cli.Arg("devices",      type=int, default=[], nargs="*"),
        ]
