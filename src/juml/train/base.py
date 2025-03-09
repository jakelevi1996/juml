import torch
from jutility import cli, util
from juml import device
from juml.models.base import Model
from juml.datasets.base import Dataset

class Trainer:
    def __init__(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
        gpu:        bool,
        **kwargs,
    ):
        raise NotImplementedError()

    @classmethod
    def from_args(
        cls,
        args:       cli.ParsedArgs,
        seed:       int,
        gpu:        bool,
        devices:    list[int],
        configs:    list[str],
    ) -> "Trainer":
        cls.apply_configs(args, configs, [])
        device.set_visible(devices)
        torch.manual_seed(seed)

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

        if gpu:
            model.cuda()
            dataset.loss.cuda()

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
            gpu=gpu,
        )
        assert isinstance(trainer, Trainer)

        return trainer

    @classmethod
    def apply_configs(
        cls,
        args:           cli.ParsedArgs,
        config_paths:   list[str],
        forbidden:      list[str],
    ):
        for cp in config_paths:
            print("Loading config from \"%s\"" % cp)
            config_dict = util.load_json(cp)
            assert isinstance(config_dict, dict)

            extra_keys = set(config_dict.keys()) & set(forbidden)
            if len(extra_keys) > 0:
                raise ValueError(
                    "Configuration file \"%s\" contains forbidden keys %s"
                    % (cp, extra_keys)
                )

            args.update(config_dict)

    @classmethod
    def init_sub_objects(
        self,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
    ):
        return

    @classmethod
    def get_model_name(cls, args: cli.ParsedArgs) -> str:
        model_name = args.get_value("model_name")
        if model_name is None:
            model_name = "d%s_m%s_t%s" % tuple(
                a.get_value_summary() + a.get_summary()
                for a in [
                    args.get_arg(name)
                    for name in ["dataset", "model", "trainer"]
                ]
            )

        return model_name

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
