import os
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir(
    "test_commands",
    "test_plotconfusionmatrix",
)

def test_plotconfusionmatrix():
    printer = util.Printer("test_plotconfusionmatrix", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_plotconfusionmatrix")

    output_img_path = (
        "results/train/test_commands_test_plotconfusionmatrix_model/"
        "Confusion_matrix.png"
    )

    model_path = (
        "results/train/test_commands_test_plotconfusionmatrix_model/"
        "model.pth"
    )

    parser = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--trainer BpSp "
        "--trainer.BpSp.epochs 2 "
        "--model Mlp "
        "--model.Mlp.hidden_dim 20 "
        "--model.Mlp.num_hidden_layers 1 "
        "--model.Mlp.embedder Flatten "
        "--model.Mlp.embedder.Flatten.n 3 "
        "--dataset RandomImage "
        "--model_name test_commands_test_plotconfusionmatrix_model"
    )
    args = parser.parse_args(args_str.split())
    args.get_command().run(args)
    assert os.path.isfile(model_path)

    if os.path.isfile(output_img_path):
        os.remove(output_img_path)

    assert not os.path.isfile(output_img_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "plotconfusionmatrix "
        "--model_name test_commands_test_plotconfusionmatrix_model "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.PlotConfusionMatrix)

    command.run(args)
    assert os.path.isfile(output_img_path)
