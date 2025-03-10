import torch
from jutility import util, cli
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_cli_args")

def test_cli_args_linearmodel():
    printer = util.Printer("test_cli_args_linearmodel", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_cli_args_linearmodel")

    parser = cli.Parser(
        cli.ObjectChoice(
            "model",
            juml.models.Linear.get_cli_arg(),
            juml.models.Mlp.get_cli_arg(),
            is_group=True,
        ),
        cli.ObjectChoice(
            "dataset",
            juml.datasets.Linear.get_cli_arg(),
            juml.datasets.Mnist.get_cli_arg(),
            is_group=True,
        ),
    )
    arg_str = (
        "--model Linear "
        "--dataset Linear "
        "--dataset.Linear.input_dim     13  "
        "--dataset.Linear.output_dim    7   "
        "--dataset.Linear.n_train       789 "
        "--dataset.Linear.n_test        456 "
    )
    args = parser.parse_args(arg_str.split())

    assert args.get_summary() == (
        "dLdi13dnte456dntr789do7dt0.0dx0.0mLmeImpI"
    )

    cli.verbose.set_printer(printer)
    with cli.verbose:
        dataset = args.init_object("dataset")
        assert isinstance(dataset, juml.datasets.Linear)

        model = args.init_object(
            "model",
            input_shape=dataset.get_input_shape(),
            output_shape=dataset.get_output_shape(),
        )
        assert isinstance(model, juml.models.Linear)

    data_loader = dataset.get_data_loader("train", 67)
    x, t = next(iter(data_loader))
    y = model.forward(x)

    assert repr(dataset) == "Linear(n_train=789, n_test=456)"
    assert repr(model)   == "Linear(num_params=98)"

    assert dataset.get_input_shape()  == [13]
    assert dataset.get_output_shape() == [7]

    assert isinstance(x, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert list(x.shape) == [67, 13]
    assert list(t.shape) == [67, 7]
    assert list(y.shape) == [67, 7]

    printer.hline()
    printer(parser.help())
    printer(dataset)
    printer(model)

def test_cli_args_cnn():
    printer = util.Printer("test_cli_args_cnn", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_cli_args_cnn")

    parser = cli.Parser(
        cli.ObjectChoice(
            "model",
            juml.models.Linear.get_cli_arg(),
            juml.models.Mlp.get_cli_arg(),
            juml.models.Cnn.get_cli_arg(),
            is_group=True,
        ),
        cli.ObjectChoice(
            "dataset",
            juml.datasets.Linear.get_cli_arg(),
            juml.datasets.Mnist.get_cli_arg(),
            is_group=True,
        ),
    )
    arg_str = (
        "--model Cnn "
        "--model.Cnn.pooler Average2d "
        "--dataset Mnist "
    )
    args = parser.parse_args(arg_str.split())

    assert args.get_summary() == (
        "dMmCmb2mc64meImk5mn3mpAVms2"
    )

    cli.verbose.set_printer(printer)
    with cli.verbose:
        dataset = args.init_object("dataset")
        assert isinstance(dataset, juml.base.Dataset)

        model = args.init_object(
            "model",
            input_shape=dataset.get_input_shape(),
            output_shape=dataset.get_output_shape(),
        )
        assert isinstance(model, juml.base.Model)

    assert repr(dataset) == "Mnist(n_train=60.0k, n_test=10.0k)"
    assert repr(model)   == "Cnn(num_params=617.1k)"

    data_loader = dataset.get_data_loader("train", 67)
    x, t = next(iter(data_loader))
    y = model.forward(x)
    assert list(y.shape) == [67, 10]

    printer.hline()
    printer(parser.help())
    printer(dataset)
    printer(model)
