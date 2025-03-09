from jutility import cli
from juml.commands.base import Command
from juml.train.base import Trainer

class Train(Command):
    def run(self, args: cli.ParsedArgs):
        return Trainer.from_args(args, **self.get_kwargs())

    @classmethod
    def get_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return train_args
