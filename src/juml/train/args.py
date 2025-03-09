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
                "train.dataset",
            )
            assert isinstance(dataset, Dataset)

            model = args.init_object(
                "train.model",
                input_shape=dataset.get_input_shape(),
                output_shape=dataset.get_output_shape(),
            )
            assert isinstance(model, Model)

            trainer_type = args.get_type(
                "train.trainer",
            )
            assert issubclass(trainer_type, Trainer)

            trainer_type.init_sub_objects(args, model, dataset)

        trainer = args.init_object(
            "train.trainer",
            args=args,
            model=model,
            dataset=dataset,
        )
        assert isinstance(trainer, Trainer)

    @classmethod
    def get_cli_arg(
        cls,
        models:     list[type[Model]],
        datasets:   list[type[Dataset]],
        trainers:   list[type[Trainer]],
        defaults:   dict[str, str | None],
    ) -> cli.ObjectArg:
        return cli.ObjectArg(
            cls,
            cli.ObjectChoice(
                "model",
                *[t.get_cli_arg() for t in models],
                default=defaults["model"],
                is_group=True,
            ),
            cli.ObjectChoice(
                "dataset",
                *[t.get_cli_arg() for t in datasets],
                default=defaults["dataset"],
                is_group=True,
            ),
            cli.ObjectChoice(
                "trainer",
                *[t.get_cli_arg() for t in trainers],
                default=defaults["trainer"],
                is_group=True,
            ),
            cli.NoTagArg("model_name", type=str, default=None),
            name="train",
        )
