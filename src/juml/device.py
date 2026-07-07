import os
import torch
from jutility import cli

class DeviceConfig:
    def __init__(self, visible_devices: list[int], gpu: bool):
        self.visible_devices = visible_devices
        self.gpu = gpu or (len(visible_devices) > 0)

    def set_visible_devices(self):
        if len(self.visible_devices) > 0:
            devices_str = ",".join(str(d) for d in self.visible_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = devices_str

    def set_module_device(self, module: torch.nn.Module):
        if self.gpu:
            module.cuda()

    def set_tensor_device(
        self,
        *input_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.gpu:
            input_tensors = tuple(x.cuda() for x in input_tensors)

        return input_tensors

    @classmethod
    def get_cli_arg(cls) -> cli.ObjectArg:
        return cli.ObjectArg(
            DeviceConfig,
            cli.Arg(
                "visible_devices",
                type=int,
                nargs="*",
                default=[],
                tagged=False,
            ),
            cli.Arg("gpu", action="store_true", tagged=False),
            tag="device_cfg",
            is_group=True,
        )
