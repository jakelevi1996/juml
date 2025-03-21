from juml.commands.base import Command
from juml.commands.train import Train
from juml.commands.sweep import Sweep
from juml.commands.profile import Profile
from juml.commands.plot_confusion_matrix import PlotConfusionMatrix
from juml.commands.plot_1d_regression import Plot1dRegression

def get_all() -> list[type[Command]]:
    return [
        Train,
        Sweep,
        Profile,
        PlotConfusionMatrix,
        Plot1dRegression,
    ]
