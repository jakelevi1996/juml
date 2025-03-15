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
        table:      util.Table,
        **kwargs,
    ):
        raise NotImplementedError()

    def _init_trainer(
        self,
        model:      Model,
        dataset:    Dataset,
        table:      util.Table,
    ):
        self.model      = model
        self.dataset    = dataset
        self.table      = table

    @classmethod
    def from_args(
        cls,
        args:           cli.ParsedArgs,
        seed:           int,
        devices:        list[int],
        configs:        list[str],
        printer:        util.Printer,
        print_level:    int,
    ) -> "Trainer":
        cls.apply_configs(args, configs, [])
        device.set_visible(devices)
        torch.manual_seed(seed)

        dataset = cls.init_dataset(args)
        model   = cls.init_model(args, dataset)

        gpu = (len(devices) > 0)
        if gpu:
            model.cuda()
            dataset.loss.cuda()

        trainer_type = args.get_type("trainer")
        assert issubclass(trainer_type, Trainer)
        trainer_type.init_sub_objects(args, model, dataset)

        trainer = args.init_object(
            "trainer",
            args=args,
            model=model,
            dataset=dataset,
            gpu=gpu,
            table=util.Table(
                *trainer_type.get_table_columns(),
                printer=printer,
                print_interval=util.TimeInterval(1),
                print_level=print_level,
            ),
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
    def init_dataset(
        cls,
        args: cli.ParsedArgs,
    ) -> Dataset:
        with cli.verbose:
            dataset = args.init_object("dataset")
            assert isinstance(dataset, Dataset)

        return dataset

    @classmethod
    def init_model(
        cls,
        args:       cli.ParsedArgs,
        dataset:    Dataset,
    ) -> Model:
        with cli.verbose:
            model = args.init_object(
                "model",
                input_shape=dataset.get_input_shape(),
                output_shape=dataset.get_output_shape(),
            )
            assert isinstance(model, Model)

        return model

    @classmethod
    def init_loss(
        cls,
        args:       cli.ParsedArgs,
        dataset:    Dataset,
        model:      Model,
    ) -> Loss:
        loss_arg = args.get_arg("loss")
        loss_arg.set_default_choice(  model.get_default_loss())
        loss_arg.set_default_choice(dataset.get_default_loss())

        with cli.verbose:
            loss = args.init_object("loss")
            assert isinstance(loss, Loss)

        if loss.needs_weights():
            loss.set_weights(dataset.get_loss_weights())

        return loss

    @classmethod
    def init_sub_objects(
        cls,
        args:       cli.ParsedArgs,
        model:      Model,
        dataset:    Dataset,
    ):
        raise NotImplementedError()

    def save_results(self, args: cli.ParsedArgs):
        self.output_dir = self.get_output_dir(args)
        self.model_name = os.path.basename(self.output_dir)

        time_list       = self.table.get_data("t")
        batch_loss      = self.table.get_data("batch_loss")
        train_metric    = self.table.get_data("train_metric")
        test_metric     = self.table.get_data("test_metric")

        cmd         = util.get_argv_str()
        arg_dict    = args.get_value_dict()
        metrics     = {
            "repr_model":   repr(self.model),
            "model_name":   self.model_name,
            "model_dir":    self.output_dir,
            "num_params":   self.model.num_params(),
            "time_str":     util.time_format(time_list[-1]),
            "time":         time_list[-1],
            "train":        {
                "start":    train_metric[0],
                "end":      train_metric[-1],
                "max":      max(train_metric),
                "min":      min(train_metric),
            },
            "test":         {
                "start":    test_metric[0],
                "end":      test_metric[-1],
                "max":      max(test_metric),
                "min":      min(test_metric),
            },
        }
        kw = {
            "item_fmt":     "%s = %8.5f",
            "key_order":    ["start", "end", "max", "min"],
        }
        metrics["train_summary"] = util.format_dict(metrics["train"], **kw)
        metrics[ "test_summary"] = util.format_dict(metrics["test" ], **kw)

        util.save_text(cmd,         "cmd",      self.output_dir)
        util.save_json(arg_dict,    "args",     self.output_dir)
        util.save_json(metrics,     "metrics",  self.output_dir)
        self.plot_metrics(batch_loss, train_metric, test_metric)
        torch.save(
            self.model.state_dict(),
            util.get_full_path("model.pth", self.output_dir),
        )
        self.table.save_pickle("table", self.output_dir)
        print("Model name = `%s`" % self.model_name)
        print(
            "Final metrics = %.5f (train), %.5f (test)"
            % (train_metric[-1], test_metric[-1])
        )

    @classmethod
    def get_output_dir(cls, args: cli.ParsedArgs) -> str:
        return os.path.join("results", "train", cls.get_model_name(args))

    @classmethod
    def get_model_name(cls, args: cli.ParsedArgs) -> str:
        model_name = args.get_value("model_name")
        if model_name is None:
            model_name = cls.get_summary(args)

        return model_name

    @classmethod
    def get_summary(cls, args: cli.ParsedArgs) -> str:
        return "d%s_m%s_t%s_s%s" % tuple(
            a.get_value_summary() + a.get_summary()
            for a in [
                args.get_arg(name)
                for name in ["dataset", "model", "trainer", "seed"]
            ]
        )

    def plot_metrics(
        self,
        batch_loss:     list[float],
        train_metric:   list[float],
        test_metric:    list[float],
    ):
        train_label = "Train (final = %.5f)" % train_metric[-1]
        test_label  =  "Test (final = %.5f)" %  test_metric[-1]

        mp = plotting.MultiPlot(
            plotting.Subplot(
                plotting.Line(batch_loss),
                xlabel="Batch",
                **self.dataset.loss.info(),
            ),
            plotting.Subplot(
                plotting.Line(train_metric, c="b", label=train_label),
                plotting.Line(test_metric,  c="r", label=test_label),
                plotting.Legend(),
                xlabel="Epoch",
                **self.dataset.loss.metric_info(),
            ),
            title="%r\n%r" % (self.model, self.dataset),
            title_font_size=15,
            figsize=[10, 4],
        )
        mp.save("metrics", self.output_dir)

    @classmethod
    def load(cls, args: cli.ParsedArgs) -> tuple[str, Model, Dataset]:
        model_dir = cls.get_output_dir(args)

        args_path = util.get_full_path("args.json", model_dir, loading=True)
        args_dict = util.load_json(args_path)
        args.update(args_dict, allow_new_keys=True)

        dataset = cls.init_dataset(args)
        model   = cls.init_model(args, dataset)

        model_path = util.get_full_path("model.pth", model_dir, loading=True)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return model_dir, model, dataset

    @classmethod
    def get_metrics_path(cls, args: cli.ParsedArgs) -> str:
        return os.path.join(cls.get_output_dir(args), "metrics.json")

    @classmethod
    def get_table_columns(cls) -> list[util.Column]:
        raise NotImplementedError()

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
