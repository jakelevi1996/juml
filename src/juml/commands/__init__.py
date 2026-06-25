from juml.commands.command import Command
from juml.commands.train_classification import TrainClassification

def get_all_commands() -> list[type[Command]]:
    return [
        TrainClassification,
    ]
