import torch
from jutility import util, cli
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_cli_args")

def test_cli_args():
    printer = util.Printer("test_cli_args", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_cli_args")

    parser = cli.Parser(
        cli.ObjectChoice(
            "model",
            juml.models.LinearModel.get_cli_arg(),
            juml.models.Mlp.get_cli_arg(),
            default="LinearModel",
            is_group=True,
        ),
        cli.ObjectChoice(
            "dataset",
            juml.datasets.Linear.get_cli_arg(),
            juml.datasets.Mnist.get_cli_arg(),
            default="Linear",
            is_group=True,
        ),
    )
    arg_str = (
        "--dataset.Linear.input_dim     13  "
        "--dataset.Linear.output_dim    7   "
        "--dataset.Linear.n_train       789 "
        "--dataset.Linear.n_test        456 "
    )
    args = parser.parse_args(arg_str.split())

    assert args.get_summary() == (
        "dLd.i13d.nte456d.ntr789d.o7d.t0.0d.x0.0mLm.eIm.pI"
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
        assert isinstance(model, juml.models.LinearModel)

    optimiser = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    data_loader = dataset.get_data_loader("train", 67)
    x, t = next(iter(data_loader))
    y1 = model.forward(x)
    loss1 = dataset.loss.forward(y1, t)

    optimiser.zero_grad()
    loss1.backward()
    optimiser.step()

    y2 = model.forward(x)
    loss2 = dataset.loss.forward(y2, t)

    assert repr(dataset) == "Linear(n_train=789, n_test=456)"
    assert repr(model)   == "LinearModel(num_params=98)"

    assert dataset.get_input_shape()  == [13]
    assert dataset.get_output_shape() == [7]

    assert isinstance(x,        torch.Tensor)
    assert isinstance(t,        torch.Tensor)
    assert isinstance(y1,       torch.Tensor)
    assert isinstance(y2,       torch.Tensor)
    assert isinstance(loss1,    torch.Tensor)
    assert isinstance(loss2,    torch.Tensor)

    assert list(x.shape)        == [67, 13]
    assert list(t.shape)        == [67, 7]
    assert list(y1.shape)       == [67, 7]
    assert list(y2.shape)       == [67, 7]
    assert list(loss1.shape)    == []
    assert list(loss2.shape)    == []

    assert loss1.item() > 0
    assert loss2.item() > 0
    assert loss2.item() < loss1.item()

    printer.hline()
    printer(parser.help())
    printer(dataset)
    printer(model)
