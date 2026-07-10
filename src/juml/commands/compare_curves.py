from jutility import util, cli
from juml.commands.command import Command
from juml.commands.train_classification import TrainClassification
from juml.commands.sweep import Sweep
from juml.experiments.curve_group import LearningCurveGroup
from juml.experiments.plot_config import PlottingConfig

class CompareLearningCurves(Sweep):
    def run(
        self,
        params:     dict[str, list],
        name:       str | None,
        force_run:  bool,
    ):
        command = self.get_command()
        assert isinstance(command, Command)

        cfg = self.init_object("PlottingConfig")
        assert isinstance(cfg, PlottingConfig)

        self.experiment_group = LearningCurveGroup.from_params(
            params=params,
            command=command,
            force_run=force_run,
        )

        if name is None:
            name = self.experiment_group.get_summary()

        self.name = name

        print("\nExperiments:\n")
        self.experiment_group.get_table(None)

        self.experiment_group.run()

        util.hline()
        self.save_cmd()
        self.save_args()

        output_dir = self.get_output_dir()

        if cfg.y_key is not None:
            mp = self.experiment_group.get_multiplot(cfg)
            mp.save("sweep results", output_dir)
