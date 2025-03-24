import os
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir(
    "test_commands",
    "test_plot1dregression",
)

def test_plot1dregression():
    printer = util.Printer("test_plot1dregression", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_plot1dregression")

    output_img_path = (
        "results/train/test_commands_test_plot1dregression_model/"
        "Predictions.png"
    )

    model_path = (
        "results/train/test_commands_test_plot1dregression_model/model.pth"
    )
    if os.path.isfile(model_path):
        os.remove(model_path)

    assert not os.path.isfile(model_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--trainer BpSp "
        "--trainer.BpSp.epochs 2 "
        "--model Mlp "
        "--model.Mlp.hidden_dim 11 "
        "--model.Mlp.depth 1 "
        "--dataset SinMix "
        "--model_name test_commands_test_plot1dregression_model"
    )
    args = parser.parse_args(args_str.split())
    args.get_command().run(args)

    assert os.path.isfile(model_path)

    if os.path.isfile(output_img_path):
        os.remove(output_img_path)

    assert not os.path.isfile(output_img_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "plot1dregression "
        "--model_name test_commands_test_plot1dregression_model "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Plot1dRegression)

    command.run(args)
    assert os.path.isfile(output_img_path)
