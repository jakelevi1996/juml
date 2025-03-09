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

    s = "train --model Mlp --dataset Mnist"
    args = parser.parse_args(s.split())
    assert juml.base.Trainer.get_model_name(args) == (
        "dM_mMeIh100n3pI_tBb100e10lCle1E-05oADAMol0.001_s0"
    )

    s = (
        "train "
        "--gpu "
        "--seed 999 "
        "--model Cnn "
        "--model.Cnn.kernel_size 42 "
        "--dataset Cifar10 "
        "--trainer.BpSup.batch_size 1234 "
        "--trainer.BpSup.optimiser AdamW "
        "--trainer.BpSup.optimiser.AdamW.weight_decay 6.789 "
        "--trainer.BpSup.lrs.CosineAnnealingLR.eta_min 0 "
    )
    args = parser.parse_args(s.split())
    assert juml.base.Trainer.get_model_name(args) == (
        "dC_mCb2c64eIk42n3pIs2_tBb1234e10lCle0.0oADAMWol0.001ow6.789_s999"
    )

    s = "train --model_name abcdef"
    args = parser.parse_args(s.split())
    assert juml.base.Trainer.get_model_name(args) == "abcdef"
