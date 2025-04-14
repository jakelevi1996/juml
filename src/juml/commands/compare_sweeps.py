import os
from jutility import cli, util, plotting
from juml.commands.base import Command
from juml.datasets.base import Dataset
from juml.loss.base import Loss
from juml.tools.experiment import Experiment, ExperimentGroup

class CompareSweeps(Command):
    def run(
        self,
        args:   cli.ParsedArgs,
        config: list[dict[str, str]],
        xlabel: str,
        log_x:  bool,
    ):
        dataset_type = args.get_type("dataset")
        assert issubclass(dataset_type, Dataset)

        loss_arg = args.get_arg("loss")
        loss_arg.set_default_choice(dataset_type.get_default_loss())
        loss_type = args.get_type("loss")
        assert issubclass(loss_type, Loss)

        metric_info = loss_type.metric_info()
        maximise    = loss_type.metric_higher_is_better()
        log_y       = metric_info.get("log_y", False)
        opt_str     = "max" if maximise else "min"
        cp          = plotting.ColourPicker(len(config), cyclic=False)

        series = [
            SweepSeries(
                **c,
                maximise=maximise,
                opt_str=opt_str,
                colour=cp.next(),
                log_y=log_y,
            )
            for c in config
        ]

        self.name = util.merge_strings([s.sweep_name for s in series])
        self.output_dir = os.path.join("results", "compare", self.name)

        x_index = any((not s.experiments.is_numeric()) for s in series)
        xtick_config = plotting.NoisyData(x_index=x_index)
        for s in series:
            for param_vals in s.experiments.params.values():
                for v in param_vals:
                    xtick_config.update(v, None)

        axis_kwargs = {"xlabel": xlabel, "log_x": log_x}
        axis_kwargs.update(xtick_config.get_xtick_kwargs())

        mp = plotting.MultiPlot(
            plotting.Subplot(
                *[
                    s.plot(x_index, TrainGetter())
                    for s in series
                ],
                **axis_kwargs,
                **metric_info,
                title="Best train metric",
            ),
            plotting.Subplot(
                *[
                    s.plot(x_index, TestGetter())
                    for s in series
                ],
                **axis_kwargs,
                **metric_info,
                title="Best test metric",
            ),
            legend=plotting.FigureLegend(
                *[s.get_legend_plottable() for s in series],
                num_rows=None,
                loc="outside center right",
            )
        )
        mp.save("metrics", self.output_dir)

        md = util.MarkdownPrinter("summary", self.output_dir)
        md.title("Sweep comparison")
        md.heading("Metrics")
        md.image("metrics.png")
        md.git_add(md.get_filename(), mp.full_path)
        md.readme_include("`[ compare_sweeps ]`", mp.full_path)
        md.heading("`comparesweeps` command", end="\n")
        md.code_block(util.get_argv_str())

        return mp

    @classmethod
    def include_arg(cls, arg: cli.Arg) -> bool:
        return arg.name in ["dataset", "loss"]

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.JsonArg(
                "config",
                required=True,
                metavar=": list[dict[\"series_name\": str, "
                "\"sweep_name\": str, \"param_name\": str]]",
                help=(
                    "EG '[{\"series_name\":\"MLP\","
                    "\"sweep_name\":\"mlp_sweep_name_s1,2,3\","
                    "\"param_name\":\"model.RzMlp.depth\"},"
                    "{\"series_name\":\"CNN\","
                    "\"sweep_name\":\"cnn_sweep_name_s1,2,3\","
                    "\"param_name\":\"model.RzCnn.depth\"}]'"
                )
            ),
            cli.Arg("xlabel",   type=str, required=True),
            cli.Arg("log_x",    action="store_true"),
        ]

class SweepSeries:
    def __init__(
        self,
        series_name:    str,
        sweep_name:     str,
        param_name:     str,
        maximise:       bool,
        opt_str:        str,
        colour:         list[float],
        log_y:          bool,
    ):
        self.series_name    = series_name
        self.sweep_name     = sweep_name
        self.param_name     = param_name
        self.opt_str        = opt_str
        self.colour         = colour
        self.log_y          = log_y

        eg_all              = ExperimentGroup.load(sweep_name)
        best                = max(eg_all) if maximise else min(eg_all)
        self.experiments    = eg_all.sweep_param(best, param_name)

    def plot(self, x_index: bool, mg: "MetricGetter") -> plotting.Plottable:
        nd = plotting.NoisyData(log_y=self.log_y, x_index=x_index)
        for e in self.experiments:
            x = e.arg_dict[self.param_name]
            y = mg.get(e, self)
            nd.update(x, y)

        return nd.plot(c=self.colour)

    def get_legend_plottable(self) -> plotting.Plottable:
        nd = plotting.NoisyData()
        return nd.plot(c=self.colour, label=self.series_name)

class MetricGetter:
    def get(self, e: Experiment, s: SweepSeries) -> float:
        raise NotImplementedError()

class TrainGetter(MetricGetter):
    def get(self, e: Experiment, s: SweepSeries) -> float:
        return e.metrics["train"][s.opt_str]

class TestGetter(MetricGetter):
    def get(self, e: Experiment, s: SweepSeries) -> float:
        return e.metrics["test"][s.opt_str]

class TimeGetter(MetricGetter):
    def get(self, e: Experiment, s: SweepSeries) -> float:
        return e.metrics["time"]

class SizeGetter(MetricGetter):
    def get(self, e: Experiment, s: SweepSeries) -> float:
        return e.metrics["num_params"]
