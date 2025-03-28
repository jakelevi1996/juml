from jutility import cli, util
from juml.commands.base import Command
from juml.train.base import Trainer

class Train(Command):
    def run(self, args: cli.ParsedArgs):
        return Trainer.from_args(
            args=args,
            printer=util.Printer(),
            **args.get_kwargs(),
        )
