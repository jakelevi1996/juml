import os
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_commands/test_sweep")

def test_sweep():
    printer = util.Printer("test_sweep", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_sweep")

    output_path = (
        "results/sweep/dLi5o7te200tr200ts0.0x0.0lMmLeIpI"
        "tBb100e2,3lCle1E-05oAol0.001s11,2/results.md"
    )
    if os.path.isfile(output_path):
        os.remove(output_path)

    assert not os.path.isfile(output_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "sweep "
        "--model LinearModel "
        "--dataset LinearDataset "
        "--dataset.LinearDataset.input_dim 5 "
        "--dataset.LinearDataset.output_dim 7 "
        "--Sweeper.seeds 2 11 "
        "--Sweeper.params {\"trainer.BpSp.epochs\":[2,3]} "
        "--Sweeper.devices [[]] "
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Sweep)

    sweeper = command.run(args)
    assert isinstance(sweeper, juml.train.Sweeper)

    printer(sweeper.name)
    assert sweeper.name == (
        "dLi5o7te200tr200ts0.0x0.0lMmLeIpItBb100e2,3lCle1E-05oAol0.001s11,2"
    )
    assert len(sweeper.experiment_dict) == 4

    assert os.path.isfile(output_path)
