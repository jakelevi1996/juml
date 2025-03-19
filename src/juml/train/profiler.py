import torch
from torch.autograd.profiler_util import FunctionEventAvg
from jutility import cli, util, units
from juml import device
from juml.train.base import Trainer

class Profiler:
    def __init__(
        self,
        args:           cli.ParsedArgs,
        batch_size:     int,
        num_warmup:     int,
        num_profile:    int,
        devices:        list[int],
    ):
        device.set_visible(devices)
        model_dir, model, dataset = Trainer.load(args)
        train_loader = dataset.get_data_loader("train", batch_size)
        x, t = next(iter(train_loader))
        gpu = (len(devices) > 0)
        if gpu:
            [x] = device.to_device([x], gpu)
            model.cuda()

        for _ in range(num_warmup):
            y = model.forward(x)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if gpu:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        profiler_kwargs = {
            "activities":       activities,
            "profile_memory":   True,
            "with_flops":       True,
        }
        with torch.profiler.profile(**profiler_kwargs) as prof:
            with torch.profiler.record_function("model.forward"):
                for _ in range(num_profile):
                    y = model.forward(x)

        self.ka = prof.key_averages()
        printer = util.Printer("profile", dir_name=model_dir)
        printer(self.ka.table(sort_by="cpu_time_total"))

        self.cpu_total  = self.get_cpu_total(self.ka)
        self.cuda_total = self.get_cuda_total(self.ka)
        self.t_total    = self.cpu_total + self.cuda_total
        self.n_samples  = batch_size * num_profile
        self.t_sample   = self.t_total / self.n_samples
        self.throughput = 1 / self.t_sample
        self.flops      = self.get_flops_total(self.ka) / self.n_samples
        profile_dict    = {
            "t_total":          self.t_total,
            "t_sample":         self.t_sample,
            "throughput":       self.throughput,
            "flops":            self.flops,
            "n_samples":        self.n_samples,
            "n_samples_str":    units.metric.format(self.n_samples),
            "batch_size":       batch_size,
            "t_total_str": (
                "%.5f s"
                % self.t_total
            ),
            "t_sample_str": (
                "%.5f ms/sample"
                % (self.t_sample * 1e3)
            ),
            "throughput_str": (
                "%s samples/second"
                % units.metric.format(self.throughput)
            ),
            "flops_str": (
                "%sFLOPS/sample"
                % units.metric.format(self.flops).upper()
            ),
        }
        util.save_json(profile_dict, "profile", model_dir)
        table = util.Table.key_value(
            printer=util.MarkdownPrinter("profile", model_dir),
        )
        table.update(k="Model",             v="`%s`" % repr(model))
        for name, dict_key in [
            ("Time (total)",            "t_total_str"),
            ("Time (average)",          "t_sample_str"),
            ("Throughput",              "throughput_str"),
            ("FLOPS",                   "flops_str"),
            ("Total number of samples", "n_samples_str"),
            ("Batch size",              "batch_size"),
        ]:
            table.update(k=name, v=profile_dict[dict_key])

    @classmethod
    def get_cpu_total(cls, event_list: list[FunctionEventAvg]) -> float:
        return sum(e.self_cpu_time_total for e in event_list) * 1e-6

    @classmethod
    def get_cuda_total(cls, event_list: list[FunctionEventAvg]) -> float:
        return sum(e.self_cuda_time_total for e in event_list) * 1e-6

    @classmethod
    def get_flops_total(cls, event_list: list[FunctionEventAvg]) -> float:
        return sum(e.flops for e in event_list)

    @classmethod
    def get_cli_arg(cls) -> list[cli.Arg]:
        return cli.ObjectArg(
            Profiler,
            cli.Arg("batch_size",   type=int, default=100),
            cli.Arg("num_warmup",   type=int, default=10),
            cli.Arg("num_profile",  type=int, default=10),
            cli.Arg("devices",      type=int, default=[], nargs="*"),
            is_group=True,
        )
