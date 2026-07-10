from jutility import util, cli
from juml.commands.command import Command
from juml.commands.train_classification import TrainClassification
from juml.experiments.group import ExperimentGroup
from juml.experiments.plot_config import PlottingConfig

class Sweep(Command):
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

        self.experiment_group = ExperimentGroup.from_params(
            params=params,
            command=command,
            force_run=force_run,
        )

        if name is None:
            name = self.experiment_group.get_summary()

        self.name = name

        print("\nExperiments:\n")
        self.experiment_group.get_table(cfg.y_key)

        self.experiment_group.run()

        if cfg.y_key is not None:
            for e in self.experiment_group:
                e.load(cfg.y_key)

        print("\nResults:\n")
        table = self.experiment_group.get_table(cfg.y_key)

        util.hline()
        self.save_cmd()
        self.save_args()

        output_dir = self.get_output_dir()
        util.save_text(str(table), "summary", output_dir, "md")
        self.experiment_group.make_git_script(output_dir)
        if cfg.y_key is not None:
            self.experiment_group.save_min_max(output_dir, cfg.y_key)

        if (cfg.x_key is not None) and (cfg.y_key is not None):
            mp = self.experiment_group.get_multiplot(cfg)
            mp.save("sweep results", output_dir)

    def get_summary(self, replaces=None) -> str:
        return self.name

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return [
            cli.JsonArg(
                "params",
                required=True,
                help=(
                    "Parameters to sweep over, as a dictionary mapping key "
                    "strings to lists of values, in a JSON string of the "
                    "form '{\"a1\": [v1, v2, ...], \"a2,a3\": [v3, v4, v5, "
                    "...], ...}'. Key strings correspond to options for the "
                    "specified subcommand (without \"--\"), and will be set "
                    "to each of the values provided, in all possible "
                    "combinations. To sweep over multiple keys "
                    "simultaneously, include both keys separated by a comma "
                    "in a single key string, EG \"a2,a3\", in which case "
                    "\"a2\" and \"a3\" will always be set to the same value. "
                    "Specify which key strings should be plotted on the x, "
                    "colour, row, and column axes using the appropriate "
                    "arguments to `PlottingConfig`."
                ),
            ),
            cli.Arg("name",         type=str, default=None),
            cli.Arg("force_run",    action="store_true"),
            PlottingConfig.get_cli_arg(),
        ]

    @classmethod
    def get_subcommands(cls) -> list[type[Command]]:
        return [
            TrainClassification,
        ]
