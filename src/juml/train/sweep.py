import os
import multiprocessing
import queue
from jutility import plotting, util, cli
from juml.datasets.base import Dataset
from juml.train.base import Trainer

class Sweeper:
    def __init__(
        self,
        args:           cli.ParsedArgs,
        params:         dict[str, list],
        sweep_devices:  list[list[int]],
        seeds:          list[int],
        target_metric:  str,
        no_cache:       bool,
        log_x:          bool,
        title:          str | None,
        devices:        list[int],
        configs:        list[str],
        **train_args,
    ):
        Trainer.apply_configs(args, configs, list(params.keys()))

        with cli.verbose:
            dataset = args.init_object("dataset")
            assert isinstance(dataset, Dataset)

        self.params = params
        self.seeds = seeds
        self.target = target_metric.split(".")
        self.init_experiment_config()
        self.init_results(None)

        mp_context = multiprocessing.get_context("spawn")

        q = mp_context.Queue()
        for e in self.experiment_list:
            q.put(e)

        p_list = [
            mp_context.Process(
                target=sweeper_subprocess,
                kwargs={
                    "args":         args,
                    "q":            q,
                    "pid":          i,
                    "devices":      d,
                    "no_cache":     no_cache,
                    "train_args":   train_args,
                },
            )
            for i, d in enumerate(sweep_devices)
        ]

        for p in p_list:
            p.start()

        for p in p_list:
            p.join()

        """
        Now:

        - Find experiment dict with best metric
        - Loop over sweep arg names:
            - Make dict[str, NoisyData]
            - Loop over values:
                - Store results in dict[str, NoisyData]
            - Plot dict[str, NoisyData] in
              results/sweep/sweep_name/arg_name.png
        - Save `results/sweep/sweep_name/results.md`, including metrics.png and
          all arg_name.png
        """
        util.hline()

        model_names = []
        for e in self.experiment_list:
            args.update(e)
            metrics = util.load_json(Trainer.get_metrics_path(args))
            self.store_result(e, metrics)
            model_names.append(metrics["model_name"])

        self.name = util.merge_strings(model_names)
        self.output_dir = os.path.join("results", "sweep", self.name)
        util.save_json(self.results_dict, "results", self.output_dir)

        util.hline()
        log_y = dataset.loss.metric_info().get("log_y", False)
        x_index = any(
            (not isinstance(val, int)) and (not isinstance(val, float))
            for val in sweep_arg_vals
        )
        results_dict = {
            "train":    plotting.NoisyData(log_y=log_y, x_index=x_index),
            "test":     plotting.NoisyData(log_y=log_y, x_index=x_index),
            "time":     plotting.NoisyData(log_y=True,  x_index=x_index),
            "size":     plotting.NoisyData(log_y=True,  x_index=x_index),
        }
        arg_summaries = []

        for args_update_dict in update_list:
            val = args_update_dict[sweep_arg_name]
            args.update(args_update_dict)
            metrics_path = Trainer.get_metrics_path(args)
            metrics = util.load_json(metrics_path)

            results_dict["train"].update(val, metrics["train"]["end"])
            results_dict["test" ].update(val, metrics["test" ]["end"])
            results_dict["time" ].update(val, metrics["time" ])
            results_dict["size" ].update(val, metrics["num_params"])
            arg_summaries.append(Trainer.get_model_name(args))

        merged_summaries = util.merge_strings(arg_summaries)
        output_dir = os.path.join("results/train_sweep", merged_summaries)
        util.save_pickle(results_dict, "results_dict", output_dir)

        if title is None:
            title = "%s\n%r" % (sweep_arg_name, sweep_arg_vals)
        else:
            title = title.replace("\\n", "\n")

        mp = plotting.MultiPlot(
            plotting.Subplot(
                results_dict["train"].plot(),
                **results_dict["train"].get_xtick_kwargs(),
                **dataset.loss.metric_info(),
                xlabel=sweep_arg_name,
                log_x=log_x,
                title="Train metric",
            ),
            plotting.Subplot(
                results_dict["test"].plot(),
                **results_dict["test"].get_xtick_kwargs(),
                **dataset.loss.metric_info(),
                xlabel=sweep_arg_name,
                log_x=log_x,
                title="Test metric",
            ),
            plotting.Subplot(
                results_dict["time"].plot(),
                **results_dict["time"].get_xtick_kwargs(),
                xlabel=sweep_arg_name,
                log_x=log_x,
                log_y=True,
                ylabel="Time (s)",
                title="Time",
            ),
            plotting.Subplot(
                results_dict["size"].plot(),
                **results_dict["size"].get_xtick_kwargs(),
                xlabel=sweep_arg_name,
                log_x=log_x,
                log_y=True,
                ylabel="Number of parameters",
                title="Number of parameters",
            ),
            title=title,
        )
        mp.save("metrics", output_dir)

        cf = util.ColumnFormatter("%-20s", sep=" = ")
        for split in ["train", "test"]:
            results = results_dict[split]
            opt_dict = {
                "Smallest": results.argmin(),
                "Largest":  results.argmax(),
            }
            for opt_type in sorted(opt_dict.keys()):
                util.hline()
                val, seed_ind, metric = opt_dict[opt_type]
                seed = seeds[seed_ind]
                args.update({sweep_arg_name: val, "seed": seed})
                metrics_path = Trainer.get_metrics_path(args)
                metrics = util.load_json(metrics_path)
                print(
                    "\n%s final %s metric = %.5f, "
                    "found with `%s = %s` and `seed = %s`"
                    % (opt_type, split, metric, sweep_arg_name, val, seed)
                )
                cf.print("Model", metrics["repr_model"])
                cf.print("Model name", metrics["model_name"])
                cf.print("Training duration", metrics["time_str"])
                for s in ["train", "test"]:
                    m = metrics[s]
                    cf.print(
                        "%-5s metrics" % s.title(),
                        "%.5f (max), %.5f (min), %.5f (final)"
                        % (m["max"], m["min"], m["end"]),
                    )

                metrics_dir = os.path.dirname(metrics_path)
                img_path = os.path.join(metrics_dir, "metrics.png")
                print("\n![](%s)" % img_path)

    def init_experiment_config(self):
        components_list = [[["seed", s]] for s in self.seeds]
        for param_name, param_vals in self.params.items():
            components_list = [
                c + p
                for c in components_list
                for p in [[[param_name, v]] for v in param_vals]
            ]

        self.experiment_list = [
            {k: v for k, v in c}
            for c in components_list
        ]
        self.experiment_dict = {
            util.format_dict(e): e
            for e in self.experiment_list
        }

    def init_results(self, results_dict: dict[str, float | None] | None):
        if results_dict is None:
            results_dict = {
                s: None
                for s in self.experiment_dict.keys()
            }

        self.results_dict = results_dict

    def store_result(self, arg_dict: dict, metrics: dict):
        arg_str = util.format_dict(arg_dict)
        if arg_str not in self.results_dict:
            raise ValueError(
                "Received invalid experiment %s, choose from %s"
                % (arg_str, sorted(self.results_dict.keys()))
            )
        if self.results_dict[arg_str] is not None:
            raise ValueError(
                "Experiment %s already has result %s"
                % (arg_str, self.results_dict[arg_str])
            )

        input_metrics = metrics
        for key in self.target:
            metrics = metrics[key]

        result = metrics
        if not isinstance(result, float):
            raise ValueError(
                "Target %s in metrics %s has type %s, expected `float`"
                % (self.target, input_metrics, type(result))
            )

        self.results_dict[arg_str] = result

    @classmethod
    def get_cli_arg(cls) -> cli.ObjectArg:
        return cli.ObjectArg(
            cls,
            cli.JsonArg(
                "params",
                default=dict(),
                metavar="`dict[str, list]`",
            ),
            cli.JsonArg(
                "sweep_devices",
                default=[[]],
                metavar="`list[list[int]]`",
            ),
            cli.Arg(
                "seeds",
                type=int,
                nargs="+",
                default=list(range(5)),
            ),
            cli.Arg("target_metric",    type=str, default="test.min"),
            cli.Arg("no_cache",         action="store_true"),
            cli.Arg("log_x",            action="store_true"),
            cli.Arg("title",            type=str, default=None),
            is_group=True,
        )

def sweeper_subprocess(
    args:       cli.ParsedArgs,
    q:          multiprocessing.Queue,
    pid:        int,
    devices:    list[int],
    no_cache:   bool,
    train_args: dict,
):
    while True:
        try:
            args_update_dict = q.get(block=False)
        except queue.Empty:
            return

        args.update(args_update_dict)
        for key in args_update_dict:
            if key in train_args:
                train_args[key] = args_update_dict[key]

        metrics_path = Trainer.get_metrics_path(args)
        if (not os.path.isfile(metrics_path)) or no_cache:
            with util.Timer(args_update_dict, hline=True):
                Trainer.from_args(
                    args,
                    devices=devices,
                    configs=[],
                    **train_args,
                )
        else:
            print("Found cached results `%s` -> skip" % metrics_path)
