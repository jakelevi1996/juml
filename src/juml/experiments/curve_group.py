from jutility import plotting
from juml.experiments.group import ExperimentGroup
from juml.experiments.plot_config import PlottingConfig

class LearningCurveGroup(ExperimentGroup):
    def get_subgroup(self, key, value) -> "LearningCurveGroup":
        return LearningCurveGroup(
            params=self.params,
            experiment_list=[
                e
                for e in self.experiment_list
                if (e.updates[key] == value)
            ],
        )

    def get_subplot(
        self,
        cfg:            PlottingConfig,
        title_parts:    list[str],
    ) -> plotting.Subplot:
        if cfg.c_key is not None:
            c_vals = self.params[cfg.c_key]
            cp = cfg.get_cp(len(c_vals))
            nc_list = [
                self.get_subgroup(cfg.c_key, c).get_series(cfg)
                for c in c_vals
            ]
            lines = [
                nc.plot(c=c, label=s)
                for nc, c, s in zip(nc_list, cp, c_vals)
            ]
        else:
            nc = self.get_series(cfg)
            lines = [nc.plot()]

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

    def get_series(self, cfg: PlottingConfig) -> plotting.NoisyCurve:
        nc = plotting.NoisyCurve(log_y=cfg.log_y)
        for e in self:
            nc.update(e.load_table_data(cfg.y_key))

        return nc
