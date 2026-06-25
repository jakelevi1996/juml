from jutility import util
from juml.commands.command import Command

class Experiment:
    def __init__(
        self,
        updates:    dict[str, str | float | int | bool],
        command:    Command,
        force_run:  bool,
    ):
        self.updates = updates
        self.command = command

        self.update_command()
        self.name = command.get_summary()
        self.skip = (command.has_metrics() and (not force_run))
        self.value = None

    def update_command(self):
        split_updates = {
            ki: v
            for k, v in self.updates.items()
            for ki in k.split(",")
        }
        self.command.update(split_updates)

    def force_skip(self):
        self.skip = True

    def run(self):
        self.update_command()
        with util.Timer(name=str(self), hline=True):
            self.command.run(**self.command.get_kwargs())

    def load(self, target_metric: str):
        self.update_command()
        self.value = self.command.load_metric(target_metric)

    def get_output_dir(self) -> str:
        self.update_command()
        return self.command.get_output_dir()

    def __str__(self) -> str:
        return "%s (%s)" % (self.name, util.format_dict(self.updates))
