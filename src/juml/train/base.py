from jutility import cli, util
from juml.models.base import Model
from juml.datasets.base import Dataset

class Trainer:
    def __init__(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
        **kwargs,
    ):
        raise NotImplementedError()

    @classmethod
    def from_args(cls, args: cli.ParsedArgs) -> "Trainer":
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

        return trainer

    @classmethod
    def init_sub_objects(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
    ):
        return

    @classmethod
    def get_cli_arg(cls) -> cli.ObjectArg:
        return cli.ObjectArg(
            cls,
            *cls.get_cli_options(),
        )

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        raise NotImplementedError()

    def __repr__(self):
        return util.format_type(type(self))
