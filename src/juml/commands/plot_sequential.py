from jutility import cli
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

        mp = plot_sequential(model, x)
        mp.save("plot_sequential", model_dir)
        return mp

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return [
            cli.Arg("batch_size",   type=int, default=100),
            cli.Arg("num_warmup",   type=int, default=10),
            cli.Arg("devices",      type=int, default=[], nargs="*"),
        ]
