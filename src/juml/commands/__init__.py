from juml.commands.command import Command
from juml.commands.sweep import Sweep
from juml.commands.compare_curves import CompareLearningCurves
from juml.commands.train_classification import TrainClassification
from juml.commands.train_rand_relu_features import TrainRandReluFeatures

def get_all_commands() -> list[type[Command]]:
    return [
        Sweep,
        CompareLearningCurves,
        TrainClassification,
        TrainRandReluFeatures,
    ]
