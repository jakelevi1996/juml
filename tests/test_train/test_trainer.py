import torch
import pytest
from jutility import util, cli
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_train/test_trainer")

def test_get_model_name():
    parser = juml.base.Framework.get_parser()

    s = "train --model Mlp --dataset Mnist"
    args = parser.parse_args(s.split())
    assert juml.base.Trainer.get_model_name(args) == (
        "dM_mMeIh100n3pI_tBb100e10lCle1E-05oADAMol0.001s0"
    )

    s = (
        "train "
        "--model Cnn "
        "--model.Cnn.kernel_size 42 "
        "--dataset Cifar10 "
        "--trainer.BpSup.seed 999 "
        "--trainer.BpSup.batch_size 1234 "
        "--trainer.BpSup.optimiser AdamW "
        "--trainer.BpSup.optimiser.AdamW.weight_decay 6.789 "
        "--trainer.BpSup.lrs.CosineAnnealingLR.eta_min 0 "
    )
    args = parser.parse_args(s.split())
    assert juml.base.Trainer.get_model_name(args) == (
        "dC_mCb2c64eIk42n3pIs2_tBb1234e10lCle0.0oADAMWol0.001ow6.789s999"
    )
