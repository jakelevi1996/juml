from jutility import cli
from juml.commands.base import Command
from juml.train.args import TrainArgs

class Train(Command):
    def run(self, args: cli.ParsedArgs):
        TrainArgs.train(args)

    @classmethod
    def get_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return train_args
