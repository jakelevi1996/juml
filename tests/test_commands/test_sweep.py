import os
import pytest
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_commands/test_sweep")

@pytest.mark.parametrize("num_processes", [1, 2])
def test_sweep(num_processes: int):
    test_name = "test_sweep_num_processes_%i" % num_processes
    printer = util.Printer(test_name, dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed(test_name)

    sweep_results_dir = (
        "results/sweep/dLi5o7te200tr200ts0.0x0.0lMmLeIpI"
        "tBb100e2,3lCle1E-05oAol0.001s11,2"
    )
    output_path         = os.path.join(sweep_results_dir, "results.md")
    subprocess_log_path = os.path.join(sweep_results_dir, "p0_log.txt")
    if os.path.isfile(output_path):
        os.remove(output_path)
    if os.path.isfile(subprocess_log_path):
        os.remove(subprocess_log_path)

    assert not os.path.isfile(output_path)
    assert not os.path.isfile(subprocess_log_path)

    devices_str = ",".join("[]" for _ in range(num_processes))
    args_str = (
        "sweep "
        "--model LinearModel "
        "--dataset LinearDataset "
        "--dataset.LinearDataset.input_dim 5 "
        "--dataset.LinearDataset.output_dim 7 "
        "--sweep.seeds 2 11 "
        "--sweep.params {\"trainer.BpSp.epochs\":[2,3]} "
        "--sweep.devices [%s] "
        % devices_str
    )
    printer(args_str)

    parser = juml.base.Framework.get_parser()
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Sweep)

    kwargs  = args.get_arg(command.name).get_kwargs()
    sweeper = command.run(args, **kwargs)
    assert isinstance(sweeper, juml.tools.Sweeper)

    printer(sweeper.name)
    assert sweeper.name == (
        "dLi5o7te200tr200ts0.0x0.0lMmLeIpItBb100e2,3lCle1E-05oAol0.001s11,2"
    )
    assert len(sweeper.experiments) == 4
    assert sweeper.best.arg_str == min(sweeper.experiments).arg_str
    assert sweeper.best.arg_str == util.format_dict(sweeper.best.arg_dict)

    assert os.path.isfile(output_path)

    if num_processes > 1:
        assert os.path.isfile(subprocess_log_path)
    else:
        assert not os.path.isfile(subprocess_log_path)
