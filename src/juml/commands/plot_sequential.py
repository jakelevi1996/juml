import os
from jutility import cli, util
from juml.commands.base import Command
from juml.device import DeviceConfig
from juml.train.base import Trainer
from juml.tools.display import plot_sequential

class PlotSequential(Command):
    def run(
        self,
        args:           cli.ParsedArgs,
        batch_size:     int,
        num_warmup:     int,
        devices:        list[int],
    ):
        device_cfg = DeviceConfig(devices)
        model_dir, model, dataset = Trainer.load(args)
        device_cfg.set_module_device(model)

        train_loader = dataset.get_data_loader("train", batch_size)
        x, t = next(iter(train_loader))
        [x] = device_cfg.to_device([x])

        for _ in range(num_warmup):
            y = model.forward(x)

        md = util.MarkdownPrinter(self.name, model_dir)
        md.title(md.code(repr(model)))
        md.set_print_to_console(True)

        mp = plot_sequential(model, x, md)
        mp.save(self.name, model_dir)

        md.set_print_to_console(False)
        md.image(self.name + ".png")
        md.heading("`git add`", end="\n")
        md.code_block(
            "\ncd %s" % model_dir,
            "git add -f %s.png" % self.name,
            "git add -f %s.md"  % self.name,
            "cd %s\n" % os.path.relpath(".", model_dir),
        )
        md.heading("`README.md` include", end="\n")
        md.file_link(md.get_filename(), "`[ %s ]`" % repr(model))

        return mp

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("batch_size",   type=int, default=100),
            cli.Arg("num_warmup",   type=int, default=10),
            cli.Arg("devices",      type=int, default=[], nargs="*"),
        ]
