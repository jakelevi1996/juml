import datetime
from jutility import plotting, util
from juml.commands.command import Command
from juml.experiments.experiment import Experiment
from juml.experiments.plot_config import PlottingConfig

class ExperimentGroup:
    def __init__(
        self,
        params:             dict[str, list],
        experiment_list:    list[Experiment],
    ):
        self.params = params
        self.experiment_list = experiment_list

    @classmethod
    def from_params(
        cls,
        params:     dict[str, list],
        command:    Command,
        force_run:  bool,
    ) -> "ExperimentGroup":
        components_list = [[]]
        for name, v_list in params.items():
            new_pairs_list = [[name, v] for v in v_list]
            components_list = [
                [*c, p]
                for c in components_list
                for p in new_pairs_list
            ]

        updates_list = [
            {k: v for [k, v] in c}
            for c in components_list
        ]

        experiment_list = []
        names = set()
        for u in updates_list:
            e = Experiment(u, command, force_run)
            experiment_list.append(e)
            if e.name in names:
                e.force_skip()
            else:
                names.add(e.name)

        return cls(params, experiment_list)

    def run(self):
        run_list = [e for e in self.experiment_list if (not e.skip)]
        n = len(run_list)
        timer = util.Timer()

        for i, e in enumerate(run_list, start=1):
            e.run()

            t_elapsed = timer.get_time_taken()
            t_total = (t_elapsed / i) * n
            t_left = t_total - t_elapsed
            t_now = datetime.datetime.today()
            t_finish = t_now + datetime.timedelta(seconds=t_left)
            print(
                "Finished %i/%i (%.2f%%)" % (i, n, 100*i/n),
                "t+ %s" % util.time_format(t_elapsed, True),
                "t- %s" % util.time_format(t_left, True),
                "Finish ~ %s" % t_finish.strftime("%Y-%m-%d %H:%M:%S"),
                sep=" | ",
            )

    def get_table(self, target_metric: str | None) -> util.Table:
        max_len = max(len(e.name) for e in self.experiment_list)
        table = util.Table(
            util.CountColumn(title="N", width=-3, start=1),
            util.Column("summary",  width=-max_len),
            util.Column("skip",     width=-5),
            *[
                util.Column(
                    k,
                    width=-max(len(k), max(len(str(v)) for v in v_list)),
                    title=k,
                )
                for k, v_list in self.params.items()
            ],
            util.Column("result", title=target_metric),
        )
        for e in self.experiment_list:
            table.update(
                summary=e.name,
                skip=e.skip,
                result=e.value,
                **e.updates,
            )

        return table

    def get_summary(self) -> str:
        names = [e.name for e in self.experiment_list]
        return util.merge_strings(names)

    def make_git_script(self, output_dir: str):
        printer = util.Printer(
            "git_add",
            dir_name=output_dir,
            file_ext="sh",
            print_to_console=False,
        )

        names = "cmd.txt args.json git_add.sh summary.md sweep_results.png"
        for s in names.split():
            printer("git add -f %s/%s" % (output_dir, s))

        all_output_dirs = [
            e.get_output_dir()
            for e in self.experiment_list
        ]
        output_dirs = sorted(set(all_output_dirs))
        for s in "metrics.json args.json".split():
            printer()
            for d in output_dirs:
                printer("git add -f %s/%s" % (d, s))

    def save_min_max(self, output_dir: str, target_metric: str):
        printer = util.Printer(
            "min_max",
            dir_name=output_dir,
            print_to_console=False,
        )

        min_val = min(e.value for e in self.experiment_list)
        max_val = max(e.value for e in self.experiment_list)
        argmin = [str(e) for e in self.experiment_list if e.value == min_val]
        argmax = [str(e) for e in self.experiment_list if e.value == max_val]

        printer("min[%s] = %s" % (target_metric, min_val))
        printer("argmin[%s] =" % (target_metric))
        for s in argmin:
            printer("- %s" % s)

        printer.hline()

        printer("max[%s] = %s" % (target_metric, max_val))
        printer("argmax[%s] =" % (target_metric))
        for s in argmax:
            printer("- %s" % s)

    def get_subgroup(self, key, value) -> "ExperimentGroup":
        return ExperimentGroup(
            params=self.params,
            experiment_list=[
                e
                for e in self.experiment_list
                if (e.updates[key] == value)
            ],
        )

    def get_multiplot(self, cfg: PlottingConfig) -> plotting.MultiPlot:
        return plotting.MultiPlot(
            *self.get_all_subplots(cfg),
            fs=cfg.figsize,
            nr=(
                len(self.params[cfg.row_key])
                if cfg.row_key is not None
                else 1
            ),
            nc=(
                len(self.params[cfg.col_key])
                if cfg.col_key is not None
                else 1
            ),
            sharex=cfg.sharex,
            sharey=cfg.sharey,
            **self.get_legend_kwargs(cfg),
        )

    def get_all_subplots(self, cfg: PlottingConfig) -> list[plotting.Subplot]:
        if cfg.row_key is not None:
            row_vals = self.params[cfg.row_key]
            subplots = [
                sp
                for r in row_vals
                for sp in self.get_subgroup(cfg.row_key, r).get_row_subplots(
                    cfg,
                    ["%s=%s" % (cfg.row_label, r)],
                )
            ]
        else:
            subplots = self.get_row_subplots(cfg, [])

        return subplots

    def get_row_subplots(
        self,
        cfg:            PlottingConfig,
        title_parts:    list[str],
    ) -> list[plotting.Subplot]:
        if cfg.col_key is not None:
            col_vals = self.params[cfg.col_key]
            subplots = [
                self.get_subgroup(cfg.col_key, c).get_subplot(
                    cfg,
                    title_parts + ["%s=%s" % (cfg.col_label, c)],
                )
                for c in col_vals
            ]
        else:
            subplots = [self.get_subplot(cfg, title_parts)]

        return subplots

    def get_subplot(
        self,
        cfg:            PlottingConfig,
        title_parts:    list[str],
    ) -> plotting.Subplot:
        if cfg.c_key is not None:
            c_vals = self.params[cfg.c_key]
            cp = cfg.get_cp(len(c_vals))
            nd_list = [
                self.get_subgroup(cfg.c_key, c).get_series(cfg)
                for c in c_vals
            ]
            lines = [
                nd.plot(c=c, label=s)
                for nd, c, s in zip(nd_list, cp, c_vals)
            ]
        else:
            nd = self.get_series(cfg)
            lines = [nd.plot()]

        title_kwargs = (
            {
                "title": ",\n".join(reversed(title_parts)),
                "title_font_size": cfg.font_size,
            }
            if (len(title_parts) > 0)
            else dict()
        )
        return plotting.Subplot(
            *lines,
            xlabel=cfg.x_label,
            ylabel=cfg.y_label,
            log_x=cfg.log_x,
            log_y=cfg.log_y,
            ylim=cfg.ylim,
            **title_kwargs,
        )

    def get_series(self, cfg: PlottingConfig) -> plotting.NoisyData:
        nd = plotting.NoisyData(log_y=cfg.log_y)
        x_vals = self.params[cfg.x_key]
        for x in x_vals:
            for e in self.get_subgroup(cfg.x_key, x):
                nd.update(x, e.value)

        return nd

    def get_legend_kwargs(self, cfg: PlottingConfig) -> dict:
        kw = dict()
        if cfg.c_key is not None:
            c_vals = self.params[cfg.c_key]
            cp = cfg.get_cp(len(c_vals))
            kw["legend"] = plotting.FigureLegend.centre_right(
                *cp.get_legend_sweeps(*c_vals, key_order=c_vals),
                title=cfg.c_label,
            )

        return kw

    def __iter__(self):
        return iter(self.experiment_list)

    def __len__(self) -> int:
        return len(self.experiment_list)
