from jutility import cli
from juml.commands.base import Command
from juml.train.args import TrainArgs

class Train(Command):
    def run(self, args: cli.ParsedArgs):
        TrainArgs.from_args(args)

    @classmethod
    def get_args(cls, trainer_arg: cli.ObjectArg) -> list[cli.Arg]:
        return [trainer_arg]
