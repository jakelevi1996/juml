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
        maximise:       bool,
        no_cache:       bool,
        log_x:          list[str],
        devices:        list[int],
        configs:        list[str],
        **train_args,
    ):
        Trainer.apply_configs(args, configs, list(params.keys()))

        with cli.verbose:
            dataset = args.init_object("dataset")
            assert isinstance(dataset, Dataset)

        self.dataset    = dataset
        self.params     = params
        self.seeds      = seeds
        self.target     = target_metric.split(".")
        self.log_x      = log_x
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

        - Save `results/sweep/sweep_name/results.md`, including metrics.png and
          all arg_name.png
        - Display all experiment configs (surrounded by `hline`) before running
        - Rename `sweep_devices` to `devices`
        - Use printers specific to each pid saved in self.output_dir
        - Display details of best seed, and also mean/std of the same config
          across random seeds
        - Add `Profile` command
        - Rename Linear classes to LinearModel and LinearDataset, and rename
          LinearLayer to Linear
        """
        util.hline()

        self.model_names = dict()
        self.all_metrics = dict()
        for arg_str, arg_dict in self.experiment_dict.items():
            args.update(arg_dict)
            metrics = util.load_json(Trainer.get_metrics_path(args))
            self.store_result(arg_str, metrics)
            self.all_metrics[arg_str] = metrics
            self.model_names[arg_str] = metrics["model_name"]

        self.name = util.merge_strings(list(self.model_names.values()))
        self.output_dir = os.path.join("results", "sweep", self.name)
        util.save_json(self.results_dict, target_metric, self.output_dir)
        util.save_json(self.model_names,  "model_names", self.output_dir)

        self.best_arg_str = (
            max(
                self.results_dict.keys(),
                key=(lambda k: self.results_dict[k]),
            )
            if maximise else
            min(
                self.results_dict.keys(),
                key=(lambda k: self.results_dict[k]),
            )
        )

        self.best_result    = self.results_dict     [self.best_arg_str]
        self.best_metrics   = self.all_metrics      [self.best_arg_str]
        self.best_arg_dict  = self.experiment_dict  [self.best_arg_str]

        best_model_dir      = self.best_metrics["model_dir"]
        best_model_rel_dir  = os.path.relpath(best_model_dir, self.output_dir)
        best_metrics_png    = os.path.join(best_model_rel_dir, "metrics.png")
        self.plot_rel_paths = [best_metrics_png]

        for param_name in self.params.keys():
            self.plot_param(param_name)

        md_printer = util.Printer(
            "results",
            dir_name=self.output_dir,
            file_ext="md",
            print_to_console=False,
        )
        sh_printer = util.Printer(
            "git_add",
            dir_name=self.output_dir,
            file_ext="sh",
            print_to_console=False,
        )

        table = util.Table.key_value(width=-40, printer=md_printer)
        table.update(key="Target metric",   value="`%s`" % target_metric)
        table.update(key="Best result",     value="`%s`" % self.best_result)
        for name, metric in [
            ("Model",                   "repr_model"),
            ("Model name",              "model_name"),
            ("Training duration",       "time_str"),
            ("Number of parameters",    "num_params"),
        ]:
            table.update(key=name, value="`%s`" % self.best_metrics[metric])

        for param_name in self.params.keys():
            table.update(
                key="`--%s`" % param_name,
                value=self.best_arg_dict[param_name],
            )

        table.update(key="`--seed`", value=self.best_arg_dict["seed"])

        sh_printer("cd %s" % self.output_dir)
        sh_printer("git add -f results.md")

        for rel_path in self.plot_rel_paths:
            md_printer("\n![](%s)"      % rel_path)
            sh_printer("git add -f %s"  % rel_path)

        sh_printer("cd %s" % os.path.relpath(os.getcwd(), self.output_dir))

        md_printer.flush()
        sh_printer.flush()

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

    def store_result(self, arg_str: str, metrics: dict):
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

    def plot_param(self, param_name: str):
        param_vals  = self.params[param_name]
        best_val    = self.best_arg_dict[param_name]
        best_seed   = self.best_arg_dict["seed"]

        metric_info = self.dataset.loss.metric_info()
        log_y       = metric_info.get("log_y", False)
        log_x       = (True if (param_name in self.log_x) else False)

        x_index = any(
            (not isinstance(v, int)) and (not isinstance(v, float))
            for v in param_vals
        )
        results_dict = {
            "train":    plotting.NoisyData(log_y=log_y, x_index=x_index),
            "test":     plotting.NoisyData(log_y=log_y, x_index=x_index),
            "time":     plotting.NoisyData(log_y=True,  x_index=x_index),
            "size":     plotting.NoisyData(log_y=True,  x_index=x_index),
        }

        for v in param_vals:
            for s in self.seeds:
                self.best_arg_dict[param_name]  = v
                self.best_arg_dict["seed"]      = s
                arg_str = util.format_dict(self.best_arg_dict)
                metrics = self.all_metrics[arg_str]

                results_dict["train"].update(v, metrics["train"]["end"])
                results_dict["test" ].update(v, metrics["test" ]["end"])
                results_dict["time" ].update(v, metrics["time" ])
                results_dict["size" ].update(v, metrics["num_params"])

        self.best_arg_dict[param_name]  = best_val
        self.best_arg_dict["seed"]      = best_seed

        mp = plotting.MultiPlot(
            plotting.Subplot(
                results_dict["train"].plot(),
                **results_dict["train"].get_xtick_kwargs(),
                **metric_info,
                xlabel=param_name,
                log_x=log_x,
                title="Train metric",
            ),
            plotting.Subplot(
                results_dict["test"].plot(),
                **results_dict["test"].get_xtick_kwargs(),
                **metric_info,
                xlabel=param_name,
                log_x=log_x,
                title="Test metric",
            ),
            plotting.Subplot(
                results_dict["time"].plot(),
                **results_dict["time"].get_xtick_kwargs(),
                xlabel=param_name,
                log_x=log_x,
                log_y=True,
                ylabel="Time (s)",
                title="Time",
            ),
            plotting.Subplot(
                results_dict["size"].plot(),
                **results_dict["size"].get_xtick_kwargs(),
                xlabel=param_name,
                log_x=log_x,
                log_y=True,
                ylabel="Number of parameters",
                title="Number of parameters",
            ),
            title="%s\n%r" % (param_name, param_vals),
        )
        full_path = mp.save(param_name, self.output_dir)

        rel_path = os.path.relpath(full_path, self.output_dir)
        self.plot_rel_paths.append(rel_path)

    @classmethod
    def get_cli_arg(cls) -> cli.ObjectArg:
        return cli.ObjectArg(
            cls,
            cli.JsonArg(
                "params",
                default=dict(),
                metavar=": dict[str, list] = \"{}\"",
                help=(
                    "EG '{\"trainer.BpSp.epochs\":[100,200,300],"
                    "\"trainer.BpSp.optimiser.Adam.lr\":"
                    "[1e-5,1e-4,1e-3,1e-2]}'"
                )
            ),
            cli.JsonArg(
                "sweep_devices",
                default=[[]],
                metavar=": list[list[int]] = \"[[]]\"",
                help="EG \"[[1,2,3],[3,4],[5]]\" or \"[[],[],[],[],[],[]]\""
            ),
            cli.Arg(
                "seeds",
                type=int,
                nargs="+",
                default=list(range(5)),
            ),
            cli.Arg("target_metric",    type=str, default="test.min"),
            cli.Arg("maximise",         action="store_true"),
            cli.Arg("no_cache",         action="store_true"),
            cli.Arg("log_x",            type=str, default=[], nargs="+"),
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
