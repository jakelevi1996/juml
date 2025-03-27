import os
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_commands/test_profile")

def test_profile():
    printer = util.Printer("test_profile", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_profile")

    profile_results_path = (
        "results/train/test_commands_test_profile_model/profile.json"
    )

    model_path = "results/train/test_commands_test_profile_model/model.pth"
    if not os.path.isfile(model_path):
        parser = juml.base.Framework.get_parser()
        args_str = (
            "train "
            "--trainer BpSp "
            "--trainer.BpSp.epochs 2 "
            "--model Mlp "
            "--model.Mlp.hidden_dim 11 "
            "--model.Mlp.depth 1 "
            "--dataset SinMix "
            "--model_name test_commands_test_profile_model"
        )
        args = parser.parse_args(args_str.split())
        args.get_command().run(args)

    assert os.path.isfile(model_path)

    if os.path.isfile(profile_results_path):
        os.remove(profile_results_path)

    assert not os.path.isfile(profile_results_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "profile "
        "--model_name test_commands_test_profile_model "
        "--profile.num_warmup 3 "
        "--profile.num_profile 7 "
        "--profile.batch_size 11 "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Profile)

    kwargs = args.get_arg(command.name).get_kwargs()
    profiler = command.run(args, **kwargs)
    assert isinstance(profiler, juml.train.Profiler)
    assert isinstance(profiler.flops, float)
    assert profiler.flops > 0

    assert os.path.isfile(profile_results_path)
