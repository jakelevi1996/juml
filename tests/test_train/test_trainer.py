import os
import torch
import pytest
from jutility import util, cli
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_train/test_trainer")

def test_apply_configs():
    c1_path = util.save_json({"a": 67, "b.c": 8.9},         "c1", OUTPUT_DIR)
    c2_path = util.save_json({"a": -2, "d.efg.hi": "uvw"},  "c2", OUTPUT_DIR)
    c3_path = util.save_json({"a": -4},                     "c3", OUTPUT_DIR)
    c4_path = util.save_json({"extra_key": -4},             "c4", OUTPUT_DIR)

    parser = cli.Parser(
        cli.Arg("a",        type=int,   default=3),
        cli.Arg("b.c",      type=float, default=4.5),
        cli.Arg("d.efg.hi", type=str,   default="xyz"),
    )
    args = parser.parse_args([])
    assert args.get_value_dict() == {
        "a": 3,
        "b.c": 4.5,
        "d.efg.hi": "xyz",
    }

    juml.base.Trainer.apply_configs(args, [c1_path], [])
    assert args.get_value_dict() == {
        "a": 67,
        "b.c": 8.9,
        "d.efg.hi": "xyz",
    }

    juml.base.Trainer.apply_configs(args, [c2_path, c3_path], [])
    assert args.get_value_dict() == {
        "a": -4,
        "b.c": 8.9,
        "d.efg.hi": "uvw",
    }

    juml.base.Trainer.apply_configs(args, [c2_path], ["b.c"])
    assert args.get_value_dict() == {
        "a": -2,
        "b.c": 8.9,
        "d.efg.hi": "uvw",
    }

    with pytest.raises(ValueError):
        juml.base.Trainer.apply_configs(args, [c3_path], ["a"])
    with pytest.raises(ValueError):
        juml.base.Trainer.apply_configs(args, [c4_path], [])
    with pytest.raises(FileNotFoundError):
        juml.base.Trainer.apply_configs(args, ["not a path"], [])

def test_get_model_name():
    parser = juml.base.Framework.get_parser()

    args_str = "train --model Mlp --dataset Mnist --loss CrossEntropy"
    args = parser.parse_args(args_str.split())
    assert juml.base.Trainer.get_model_name(args) == (
        "dM_lC_mMeIh100n3pI_tBb100e10lCle1E-05oAol0.001_s0"
    )

    args_str = (
        "train "
        "--seed 999 "
        "--model Cnn "
        "--model.Cnn.kernel_size 42 "
        "--dataset Cifar10 "
        "--loss CrossEntropy "
        "--trainer.BpSp.batch_size 1234 "
        "--trainer.BpSp.optimiser AdamW "
        "--trainer.BpSp.optimiser.AdamW.weight_decay 6.789 "
        "--trainer.BpSp.lrs.CosineAnnealingLR.eta_min 0 "
    )
    args = parser.parse_args(args_str.split())
    assert juml.base.Trainer.get_model_name(args) == (
        "dC_lC_mCb2c64eIk42n3pIs2_tBb1234e10lCle0.0oAWol0.001ow6.789_s999"
    )

    args_str = "train --model_name abcdef"
    args = parser.parse_args(args_str.split())
    assert juml.base.Trainer.get_model_name(args) == "abcdef"

def test_load():
    printer = util.Printer("test_load", dir_name=OUTPUT_DIR)

    output_path = (
        "results/train/dSh1i1o1te200tr200ts0.1x0.0_lM_mMeIh11n1pI_"
        "tBb100e2lCle1E-05oAol0.001_s0/model.pth"
    )
    if os.path.isfile(output_path):
        os.remove(output_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--loss Mse "
        "--model Mlp "
        "--model.Mlp.hidden_dim 11 "
        "--model.Mlp.num_hidden_layers 1 "
        "--dataset SinMix "
        "--trainer.BpSp.epochs 2 "
    )
    args = parser.parse_args(args_str.split())

    with pytest.raises(FileNotFoundError):
        juml.base.Trainer.load(args)

    assert not os.path.isfile(output_path)
    args.get_command().run(args)
    assert os.path.isfile(output_path)

    model_dir, model, dataset = juml.base.Trainer.load(args)
    assert isinstance(model_dir,    str)
    assert isinstance(model,        juml.models.Mlp)
    assert isinstance(dataset,      juml.datasets.SinMix)
    assert os.path.isdir(model_dir)
    assert model_dir in output_path
