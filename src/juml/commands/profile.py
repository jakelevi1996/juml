import torch
from jutility import cli, util
from juml.commands.base import Command
from juml.train.base import Trainer

class Profile(Command):
    def run(self, args: cli.ParsedArgs):
        batch_size = args.get_value("batch_size")
        model_dir, model, dataset = Trainer.load(args)
        train_loader = dataset.get_data_loader("train", batch_size)
        x, t = next(iter(train_loader))

        for _ in range(args.get_value("num_warmup")):
            y = model.forward(x)

        profiler_kwargs = {
            "activities":       [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            "profile_memory":   True,
            "with_flops":       True,
        }
        with torch.profiler.profile(**profiler_kwargs) as prof:
            with torch.profiler.record_function("model.forward"):
                for _ in range(args.get_value("num_profile")):
                    y = model.forward(x)

        printer = util.Printer("profile", dir_name=model_dir)
        printer(prof.key_averages().table(sort_by="cpu_time_total"))

    @classmethod
    def get_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return [
            *train_args,
            cli.Arg("batch_size",   type=int, default=100),
            cli.Arg("num_warmup",   type=int, default=10),
            cli.Arg("num_profile",  type=int, default=10),
        ]
