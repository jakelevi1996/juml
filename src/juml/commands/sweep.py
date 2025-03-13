from jutility import cli
from juml.commands.base import Command
from juml.train.sweep import Sweeper

class Sweep(Command):
    def run(self, args: cli.ParsedArgs):
        kwargs  = self.get_kwargs()
        devices = kwargs.pop("devices")
        if len(devices) > 0:
            raise ValueError(
                "Received `--devices %s`, use `--Sweeper.devices` instead "
                "with `sweep` command"
                % devices
            )

        with cli.verbose:
            sweeper = args.init_object(
                "Sweeper",
                args=args,
                **kwargs,
            )

        return sweeper

    @classmethod
    def get_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return [
            *train_args,
            Sweeper.get_cli_arg(),
        ]
