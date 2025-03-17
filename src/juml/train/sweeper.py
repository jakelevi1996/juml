import os
import multiprocessing
import queue
import statistics
from jutility import plotting, util, cli
from juml.train.base import Trainer

class Sweeper:
    def __init__(
        self,
        args:           cli.ParsedArgs,
        params:         dict[str, list],
        devices:        list[list[int]],
        seeds:          list[int],
        target_metric:  str,
        maximise:       bool,
        no_cache:       bool,
        log_x:          list[str],
        configs:        list[str],
        **train_args,
    ):
        printer = util.Printer()
        printer.heading("Sweeper: Initialise experiments")

        Trainer.apply_configs(args, configs, list(params.keys()))
        dataset     = Trainer.init_dataset(args)
        self.loss   = Trainer.init_loss(args, dataset)

        self.params         = params
        self.seeds          = seeds
        self.target_str     = target_metric
        self.target_list    = target_metric.split(".")
        self.maximise       = maximise
        self.log_x          = log_x
        self.init_experiment_config()
        self.init_results(None)
        self.init_name(args)
        self.init_output_dir()

        printer.heading("Sweeper: Run experiments")

        self.run(
            args=args,
            devices=devices,
            no_cache=no_cache,
            train_args=train_args,
        )

        printer.heading("Sweeper: display results")

        self.all_metrics = dict()
        for arg_str, arg_dict in self.experiment_dict.items():
            args.update(arg_dict)
            metrics = util.load_json(Trainer.get_metrics_path(args))
            self.store_result(arg_str, metrics)
            self.all_metrics[arg_str] = metrics

        util.save_json(self.results_dict, self.target_str, self.output_dir)

        self.best_arg_str = (
            max(
                self.results_dict.keys(),
                key=(lambda k: self.results_dict[k]),
            )
            if self.maximise else
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

        self.save_results_markdown(best_model_rel_dir)

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

    def init_name(self, args: cli.ParsedArgs):
        self.model_names = dict()
        original_args = {k: args.get_value(k) for k in self.params.keys()}
        original_args["seed"] = args.get_value("seed")
        for i, (arg_str, arg_dict) in enumerate(
            self.experiment_dict.items(),
            start=1,
        ):
            print("(%2i) %s" % (i, arg_str))
            args.update(arg_dict)
            self.model_names[arg_str] = Trainer.get_summary(args)

        args.update(original_args)
        self.name = util.merge_strings(sorted(self.model_names.values()))

    def init_output_dir(self):
        self.output_dir = os.path.join("results", "sweep", self.name)

    def run(
        self,
        args:       cli.ParsedArgs,
        devices:    list[list[int]],
        no_cache:   bool,
        train_args: dict,
    ):
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
                    "output_dir":   self.output_dir,
                },
            )
            for i, d in enumerate(devices)
        ]

        for p in p_list:
            p.start()

        for p in p_list:
            p.join()

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
        for key in self.target_list:
            metrics = metrics[key]

        result = metrics
        if not isinstance(result, float):
            raise ValueError(
                "Target %s in metrics %s has type %s, expected `float`"
                % (self.target_list, input_metrics, type(result))
            )

        self.results_dict[arg_str] = result

    def plot_param(self, param_name: str):
        param_vals  = self.params[param_name]
        best_val    = self.best_arg_dict[param_name]
        best_seed   = self.best_arg_dict["seed"]

        metric_info = self.loss.metric_info()
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
                title="Final train metric",
            ),
            plotting.Subplot(
                results_dict["test"].plot(),
                **results_dict["test"].get_xtick_kwargs(),
                **metric_info,
                xlabel=param_name,
                log_x=log_x,
                title="Final test metric",
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
            title_font_size=15,
            figsize=[10, 8],
        )
        full_path = mp.save(param_name, self.output_dir)

        rel_path = os.path.relpath(full_path, self.output_dir)
        self.plot_rel_paths.append(rel_path)

    def save_results_markdown(self, best_model_rel_dir: str):
        md = util.MarkdownPrinter("results", self.output_dir)
        md.title("Sweep results", end="\n")

        md.heading("Summary")
        table = util.Table.key_value(printer=md)
        table.update(k="`# experiments`",   v=md.code(len(self)))
        table.update(k="Target metric",     v=md.code(self.target_str))
        table.update(k="Best result",       v=md.code(self.best_result))
        table.update(k="Best params/seed",  v=md.code(self.best_arg_str))

        for name, metric in [
            ("Model",                   "repr_model"),
            ("Model name",              "model_name"),
            ("Train metrics",           "train_summary"),
            ("Test metrics",            "test_summary"),
            ("Training duration",       "time_str"),
            ("Number of parameters",    "num_params"),
        ]:
            table.update(k=name, v=md.code(self.best_metrics[metric]))

        for name, val in self.best_arg_dict.items():
            table.update(k="`--%s`" % name, v=md.code(val))

        best_seed = self.best_arg_dict["seed"]
        seeds_results_list = []
        for s in self.seeds:
            self.best_arg_dict["seed"] = s
            seed_arg_str = util.format_dict(self.best_arg_dict)
            seed_result  = self.results_dict[seed_arg_str]
            seeds_results_list.append(seed_result)

        self.best_arg_dict["seed"] = best_seed
        mean_config = statistics.mean(seeds_results_list)
        mean_all    = statistics.mean(self.results_dict.values())
        table.update(k="Mean (best params)",    v=md.code(mean_config))
        table.update(k="Mean (all)",            v=md.code(mean_all))

        if len(self.seeds) >= 2:
            std_config  = statistics.stdev(seeds_results_list)
            std_all     = statistics.stdev(self.results_dict.values())
            table.update(k="STD (best params)", v=md.code(std_config))
            table.update(k="STD (all)",         v=md.code(std_all))

        md.heading("Sweep configuration")
        table = util.Table.key_value(md)
        for param_name, param_vals in self.params.items():
            table.update(k=md.code(param_name), v=md.code(str(param_vals)))

        table.update(k="`seeds`", v=md.code(str(self.seeds)))

        md.heading("Metrics", end="\n")
        best_metrics_json = os.path.join(best_model_rel_dir, "metrics.json")
        md.file_link(best_metrics_json, "Best metrics (JSON)")
        git_add_images = []
        for rel_path in self.plot_rel_paths:
            md.image(rel_path)
            git_add_images.append("git add -f %s" % rel_path)

        md.heading("All results")
        table = util.Table(
            util.Column("rank",     "i",    width=-10),
            util.Column("metric",   ".5f",  title=md.code(self.target_str)),
            util.Column("seed",     "i",    title="`seed`"),
            *[
                util.Column(param_name, title=md.code(param_name))
                for param_name in self.params.keys()
            ],
            util.Column("model_name"),
            printer=md,
        )
        sorted_names = sorted(
            self.results_dict.keys(),
            key=(lambda k: self.results_dict[k]),
            reverse=(True if self.maximise else False),
        )
        for i, arg_str in enumerate(sorted_names, start=1):
            table.update(
                rank=i,
                metric=self.results_dict[arg_str],
                model_name=md.code(self.model_names[arg_str]),
                **self.experiment_dict[arg_str],
            )

        md.heading("`git add`", end="\n")
        md.code_block(
            "cd %s"         % self.output_dir,
            "git add -f %s" % "results.md",
            "git add -f %s" % best_metrics_json,
            *git_add_images,
            "cd %s" % os.path.relpath(os.getcwd(), self.output_dir),
        )
        rm_path = os.path.relpath("README.md", self.output_dir)

        md.heading("%s include" % md.make_link(rm_path, "`README.md`"), "\n")
        md.file_link(md.get_filename(), "`[ full_sweep_results ]`")

        md.heading("`sweep` command", end="\n")
        md.code_block(util.get_argv_str())

        md.flush()

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
                "devices",
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

    def __len__(self):
        return len(self.experiment_list)

def sweeper_subprocess(
    args:       cli.ParsedArgs,
    q:          multiprocessing.Queue,
    pid:        int,
    devices:    list[int],
    no_cache:   bool,
    train_args: dict,
    output_dir: str,
):
    printer = util.Printer(
        "p%i_log" % pid,
        dir_name=output_dir,
        print_to_console=False,
    )
    printer("Devices = %s" % devices)
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
            with util.Timer(
                name=str(args_update_dict),
                printer=printer,
                hline=True,
            ):
                Trainer.from_args(
                    args,
                    devices=devices,
                    configs=[],
                    printer=printer,
                    **train_args,
                )
        else:
            print("Found cached results `%s` -> skip" % metrics_path)
