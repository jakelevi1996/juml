from jutility import cli
from juml.commands.base import Command
from juml.train.trainer import Trainer

class Train(Command):
    def run(self, args: cli.ParsedArgs):
        Trainer.from_args(args)

    @classmethod
    def get_args(cls, trainer_arg: cli.ObjectArg) -> list[cli.Arg]:
        return [trainer_arg]
