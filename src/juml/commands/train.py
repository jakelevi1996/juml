from jutility import cli
from juml.commands.base import Command
from juml.train.args import TrainArgs

class Train(Command):
    def run(self, args: cli.ParsedArgs):
        TrainArgs.train(args)

    @classmethod
    def get_args(cls, train_arg: cli.ObjectArg) -> list[cli.Arg]:
        return [train_arg]
