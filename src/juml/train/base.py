import os
import torch
from jutility import cli, util, plotting
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
        cls,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
    ):
        return

    @classmethod
    def save_results(
        cls,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
        table:      util.Table,
    ):
        output_dir = cls.get_output_dir(args)

        time_list    = table.get_data("t")
        batch_loss   = table.get_data("batch_loss")
        train_metric = table.get_data("train_metric")
        test_metric  = table.get_data("test_metric")

        cls.plot_metrics(
            batch_loss=batch_loss,
            train_metric=train_metric,
            test_metric=test_metric,
            dataset=dataset,
            plot_name="metrics",
            output_dir=output_dir,
            title=repr(model),
        )

        cmd         = util.get_argv_str()
        model_path  = util.get_full_path("model.pth", output_dir)
        arg_dict    = args.get_value_dict()
        metrics     = {
            "repr_model":   repr(model),
            "num_params":   model.num_params(),
            "time_str":     util.time_format(time_list[-1]),
            "time":         time_list[-1],
            "train":        train_metric,
            "test":         test_metric,
        }

        util.save_text(cmd,         "cmd",      output_dir)
        util.save_json(arg_dict,    "args",     output_dir)
        util.save_json(metrics,     "metrics",  output_dir)
        torch.save(model.state_dict(), model_path)
        table.save_pickle("table", output_dir)
        print(
            "Final metrics = %.5f (train), %.5f (test)"
            % (train_metric[-1], test_metric[-1])
        )

    @classmethod
    def get_output_dir(
        cls,
        args:            cli.ParsedArgs,
        experiment_name: str="train",
    ) -> str:
        model_name = cls.get_model_name(args)
        print("Model name = `%s`" % model_name)
        return os.path.join(".", "results", experiment_name, model_name)

    @classmethod
    def get_model_name(cls, args: cli.ParsedArgs) -> str:
        model_name = args.get_value("model_name")
        if model_name is None:
            model_name = "d%s_m%s_t%s_s%s" % tuple(
                a.get_value_summary() + a.get_summary()
                for a in [
                    args.get_arg(name)
                    for name in ["dataset", "model", "trainer", "seed"]
                ]
            )

        return model_name

    @classmethod
    def plot_metrics(
        cls,
        batch_loss:     list[float],
        train_metric:   list[float],
        test_metric:    list[float],
        dataset:        Dataset,
        plot_name:      str,
        output_dir:     str,
        **kwargs,
    ):
        kwargs.setdefault("title", plot_name)
        kwargs.setdefault("figsize", [10, 4])

        train_label = "Train (final = %.5f)" % train_metric[-1]
        test_label  =  "Test (final = %.5f)" %  test_metric[-1]

        mp = plotting.MultiPlot(
            plotting.Subplot(
                plotting.Line(batch_loss),
                xlabel="Batch",
                **dataset.loss.info(),
            ),
            plotting.Subplot(
                plotting.Line(train_metric, c="b", label=train_label),
                plotting.Line(test_metric,  c="r", label=test_label),
                plotting.Legend(),
                xlabel="Epoch",
                **dataset.loss.metric_info(),
            ),
            **kwargs,
        )
        mp.save(plot_name, output_dir)

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
