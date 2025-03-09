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

    @classmethod
    def train(cls, args: cli.ParsedArgs):
        with cli.verbose:
            dataset = args.init_object(
                "dataset",
            )
            assert isinstance(dataset, Dataset)

            model = args.init_object(
                "model",
                input_shape=dataset.get_input_shape(),
                output_shape=dataset.get_output_shape(),
            )
            assert isinstance(model, Model)

            trainer_type = args.get_type(
                "trainer",
            )
            assert issubclass(trainer_type, Trainer)

            trainer_type.init_sub_objects(args, model, dataset)

        trainer = args.init_object(
            "trainer",
            args=args,
            model=model,
            dataset=dataset,
        )
        assert isinstance(trainer, Trainer)
