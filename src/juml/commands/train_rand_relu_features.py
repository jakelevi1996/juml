import torch
from jutility import plotting, util, cli
from juml.commands.command import Command
from juml.data import ClassificationDataset, get_all_datasets
from juml.models.rand_relu_features import RandReluFeatures
from juml.util import (
    multiclass_acc,
    batched_multiclass_acc,
)
from juml.device import DeviceConfig

class TrainRandReluFeatures(Command):
    def run(
        self,
        seed:       int,
        batch_size: int,
        save_model: bool,
    ):
        torch.manual_seed(seed)

        dataset = self.init_object("dataset")
        assert isinstance(dataset, ClassificationDataset)

        model = self.init_object(
            "RandReluFeatures",
            input_dim=dataset.get_input_dim(),
            output_dim=dataset.get_output_dim(),
        )
        assert isinstance(model, RandReluFeatures)

        device_cfg = self.init_object("DeviceConfig")
        assert isinstance(device_cfg, DeviceConfig)

        device_cfg.set_module_device(model)

        train_loader = dataset.get_data_loader("train", batch_size)
        test_loader  = dataset.get_data_loader("test",  batch_size)

        table = util.Table(
            util.CountColumn(),
            util.TimeColumn(),
            util.Column("train_acc",    ".5f",  width=10),
            util.Column("test_acc",     ".5f",  width=10),
        )

        x, t = next(iter(train_loader))
        x, t = device_cfg.set_tensor_device(x, t)
        x, t = dataset.format_batch(x, t)
        y = model.fit(x, t)
        train_acc = multiclass_acc(y, t)
        test_acc=batched_multiclass_acc(
            model=model,
            data_loader=test_loader,
            dataset=dataset,
            device_cfg=device_cfg,
        )

        table.update(train_acc=train_acc, test_acc=test_acc)

        self.save_args()
        self.save_cmd()
        if save_model:
            self.save_model(model)

        t           = table.get_item("t", -1)
        n_params    = model.get_num_params()
        self.save_metrics(
            train_acc       =train_acc,
            test_acc        =test_acc,
            time            =t,
            time_str        =util.time_format(t, True),
            n_params        =n_params,
            n_params_str    =util.units.metric.format(n_params),
        )

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return [
            cli.Arg("seed",         type=int,   default=0),
            cli.Arg("batch_size",   type=int,   default=100),
            cli.Arg("save_model",   action="store_true", tagged=False),
            DeviceConfig.get_cli_arg(),
            cli.ObjectChoice(
                "dataset",
                *[
                    dataset_type.get_cli_arg()
                    for dataset_type in get_all_datasets()
                ],
                is_group=True,
            ),
            RandReluFeatures.get_cli_arg(is_group=True),
        ]
