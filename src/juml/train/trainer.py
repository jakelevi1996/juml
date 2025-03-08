from jutility import cli
from juml.models.base import Model
from juml.datasets.base import Dataset
from juml.train.modes.base import TrainMode

class Trainer:
    def __init__(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
        train_mode: TrainMode,
    ):
        self.args       = args
        self.model      = model
        self.dataset    = dataset
        self.train_mode = train_mode

    @classmethod
    def from_args(cls, args: cli.ParsedArgs):
        with cli.verbose:
            dataset = args.init_object(
                "Trainer.dataset",
            )
            assert isinstance(dataset, Dataset)

            model = args.init_object(
                "Trainer.model",
                input_shape=dataset.get_input_shape(),
                output_shape=dataset.get_output_shape(),
            )
            assert isinstance(model, Model)

            train_mode_type = args.get_type(
                "Trainer.train_mode",
            )
            assert issubclass(train_mode_type, TrainMode)

            train_mode_type.init_sub_objects(args, model, dataset)

        train_mode = args.init_object(
            "Trainer.train_mode",
            args=args,
            model=model,
            dataset=dataset,
        )
        assert isinstance(train_mode, TrainMode)

        return cls(
            args=args,
            model=model,
            dataset=dataset,
            train_mode=train_mode,
        )

    @classmethod
    def get_cli_arg(
        cls,
        models:         list[type[Model]],
        datasets:       list[type[Dataset]],
        train_modes:    list[type[TrainMode]],
        defaults:       dict[str, str | None],
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
                "train_mode",
                *[t.get_cli_arg() for t in train_modes],
                default=defaults["train_mode"],
                is_group=True,
            ),
        )
