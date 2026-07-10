from juml.commands.command import Command
from juml.commands.train_classification import TrainClassification
from juml.commands.sweep import Sweep
from juml.commands.compare_curves import CompareLearningCurves

def get_all_commands() -> list[type[Command]]:
    return [
        TrainClassification,
        Sweep,
        CompareLearningCurves,
    ]
