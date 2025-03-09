from jutility import cli
from juml.models.base import Model
from juml.datasets.base import Dataset
from juml.train.base import Trainer

class TrainArgs:
    def __init__(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
        trainer:    Trainer,
    ):
        self.args       = args
        self.model      = model
        self.dataset    = dataset
        self.trainer    = trainer
