import os
from jutility import cli, util

class Command(cli.SubCommand):
    def run(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        raise NotImplementedError()

    @classmethod
    def get_subcommands(cls) -> list[type["Command"]]:
        return []

    @classmethod
    def init_framework_command(cls) -> "Command":
        return cls(
            cls.get_name(),
            *cls.get_cli_args(),
            sub_commands=cli.SubCommandGroup(
                *[c.init_framework_command() for c in cls.get_subcommands()],
            ),
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    def get_output_dir(self) -> str:
        return "results/%s/%s" % (self.get_name().lower(), self.get_summary())

    def save_args(self) -> str:
        return util.save_json(
            self.get_value_dict(),
            "args",
            self.get_output_dir(),
        )

    def save_metrics(self, **metrics) -> str:
        return util.save_json(
            metrics,
            "metrics",
            self.get_output_dir(),
        )

    def save_cmd(self) -> str:
        return util.save_text(
            util.get_argv_str().replace(" --", " \\\n    --"),
            "cmd",
            self.get_output_dir(),
        )

    def get_metrics_path(self) -> str:
        return os.path.join(self.get_output_dir(), "metrics.json")

    def has_metrics(self) -> bool:
        return os.path.isfile(self.get_metrics_path())

    def load_metric(self, name: str):
        metrics = util.load_json(self.get_metrics_path())
        if name not in metrics:
            raise ValueError(
                "Invalid metric \"%s\", choose from %s"
                % (name, sorted(metrics))
            )

        return metrics[name]

    def __repr__(self) -> str:
        return self.get_name()
