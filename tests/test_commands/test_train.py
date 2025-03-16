import os
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_commands/test_train")

def test_train():
    printer = util.Printer("test_train", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_train")

    output_path = (
        "results/train/dSh1i3o5te200tr200ts0.1x0.0_lM_mMeIh23n2pI_"
        "tBb57e7lCle1E-05oAol0.001_s0/model.pth"
    )
    if os.path.isfile(output_path):
        os.remove(output_path)

    assert not os.path.isfile(output_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--trainer BpSp "
        "--trainer.BpSp.epochs 7 "
        "--trainer.BpSp.batch_size 57 "
        "--model Mlp "
        "--model.Mlp.hidden_dim 23 "
        "--model.Mlp.num_hidden_layers 2 "
        "--dataset SinMix "
        "--dataset.SinMix.input_dim 3 "
        "--dataset.SinMix.output_dim 5 "
        "--print_level 1 "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Train)

    trainer = command.run(args)
    assert isinstance(trainer, juml.train.BpSp)

    assert os.path.isfile(output_path)
