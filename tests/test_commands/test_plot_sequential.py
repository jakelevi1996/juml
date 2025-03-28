import os
from jutility import util, plotting
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir(
    "test_commands/test_plotsequential",
)

def test_plotsequential():
    printer = util.Printer("test_plotsequential", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_plotsequential")

    output_img_path = (
        "results/train/test_commands_test_plotsequential/"
        "plotsequential.png"
    )

    model_path = (
        "results/train/test_commands_test_plotsequential/model.pth"
    )
    if os.path.isfile(model_path):
        os.remove(model_path)

    assert not os.path.isfile(model_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--trainer BpSp "
        "--trainer.BpSp.epochs 2 "
        "--model RzMlp "
        "--model.RzMlp.model_dim 11 "
        "--model.RzMlp.depth 3 "
        "--dataset SinMix "
        "--model_name test_commands_test_plotsequential"
    )
    args = parser.parse_args(args_str.split())
    args.get_command().run(args)

    assert os.path.isfile(model_path)

    if os.path.isfile(output_img_path):
        os.remove(output_img_path)

    assert not os.path.isfile(output_img_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "plotsequential "
        "--model_name test_commands_test_plotsequential "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.PlotSequential)

    kwargs = args.get_arg(command.name).get_kwargs()
    mp = command.run(args, **kwargs)
    assert isinstance(mp, plotting.MultiPlot)

    assert os.path.isfile(output_img_path)
