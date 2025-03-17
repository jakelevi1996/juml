# juml

A Judicious, Unified and extendable framework for multi-paradigm Machine Learning research, powered by [`jutility`](https://github.com/jakelevi1996/jutility) and [PyTorch](https://pytorch.org/).

> *[Judicious [adjective]: having or showing reason and good judgment in making decisions](https://dictionary.cambridge.org/dictionary/english/judicious)*

![](scripts/img/logo_black.png)

## Contents

- [juml](#juml)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Overview](#overview)
  - [Usage examples](#usage-examples)
    - [Out of the box](#out-of-the-box)
      - [Help interface](#help-interface)
      - [Train a model](#train-a-model)
      - [Profile model](#profile-model)
      - [Plot confusion matrix](#plot-confusion-matrix)
      - [Sweep over parameters](#sweep-over-parameters)
    - [Extending `juml`](#extending-juml)
  - [Citation](#citation)

## Installation

The `juml` package is available as [a Python package on PyPI](https://pypi.org/project/juml-toolkit/), and can be installed using `pip` with the following commands:

```
python -m pip install -U pip
python -m pip install -U juml-toolkit
```

Alternatively, `juml` can be installed in "editable mode" from the [GitHub repository](https://github.com/jakelevi1996/juml):

```
git clone https://github.com/jakelevi1996/juml.git
python -m pip install -U pip
python -m pip install -e ./juml
```

The `juml` package depends on [PyTorch](https://pytorch.org/). The installation instructions for PyTorch depend on which (if any) CUDA version is available, so PyTorch won't be automatically installed by `pip` when installing `juml`. Instead, please install PyTorch following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Overview

The `juml` framework defines 6 fundamental classes (and several example subclasses), available in the [`juml.base`](https://github.com/jakelevi1996/juml/blob/main/src/juml/base.py) namespace module, which are expected to be subclassed in downstream projects:

- [`juml.base.Model`](src/juml/models/base.py)
- [`juml.base.Dataset`](src/juml/datasets/base.py)
- [`juml.base.Loss`](src/juml/loss/base.py)
- [`juml.base.Trainer`](src/juml/train/base.py)
- [`juml.base.Command`](src/juml/commands/base.py)
- [`juml.base.Framework`](src/juml/framework.py)

*Coming soon: `juml.base.Environment` for RL*

## Usage examples

The `juml` framework is designed to be extended in downstream research projects, but nontheless contains enough built-in functionality to run some simple ML experiments and visualise the results from the command line (without writing any Python code). The following subsections demonstrate (1) the built-in functionality of `juml` and (2) a simple example demonstrating how to extend `juml` with a new model and dataset.

### Out of the box

#### Help interface

```sh
juml -h
juml train -h
```

#### Train a model

```sh
juml train --model Mlp --model.Mlp.embedder Flatten --model.Mlp.embedder.Flatten.n 3 --dataset Mnist --trainer.BpSp.epochs 3
```

```txt
cli: Mnist()
cli: Flatten(n=3)
cli: Identity()
cli: Mlp(embedder=Flatten(num_params=0), hidden_dim=100, input_shape=[1, 28, 28], num_hidden_layers=3, output_shape=[10], pooler=Identity(num_params=0))
cli: CrossEntropy()
cli: Adam(lr=0.001, params=<generator object Module.parameters at 0x71be9c817290>)
Time        | Epoch      | Batch      | Batch loss | Train metric | Test metric
----------- | ---------- | ---------- | ---------- | ------------ | ------------
0.0002s     |          0 |            |            |      0.12013 |      0.11540
1.7416s     |          0 |          0 |    2.33147 |              |
2.0020s     |          0 |         63 |    0.34889 |              |
3.0004s     |          0 |        360 |    0.13585 |              |
3.8155s     |          0 |        599 |    0.22922 |              |
3.8166s     |          1 |            |            |      0.95568 |      0.95320
5.6912s     |          1 |          0 |    0.21946 |              |
6.0021s     |          1 |         83 |    0.09772 |              |
7.0001s     |          1 |        368 |    0.10331 |              |
7.8203s     |          1 |        599 |    0.13859 |              |
7.8214s     |          2 |            |            |      0.97107 |      0.96400
9.5837s     |          2 |          0 |    0.05180 |              |
10.0016s    |          2 |        114 |    0.04135 |              |
11.0006s    |          2 |        397 |    0.02003 |              |
11.7184s    |          2 |        599 |    0.08072 |              |
11.7195s    |          3 |            |            |      0.97862 |      0.97010
Saving in "results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/cmd.txt"
Saving in "results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/args.json"
Saving in "results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/metrics.json"
Saving in "results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/metrics.png"
Saving in "results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/model.pth"
Saving in "results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/table.pkl"
Model name = `dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0`
Final metrics = 0.97862 (train), 0.97010 (test)
Time taken for `train` = 13.9629 seconds
```

![](results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/metrics.png)

#### Profile model

```
juml profile --model_name dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0
```

#### Plot confusion matrix

```
juml plotconfusionmatrix --model_name dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0
```

![](results/train/dM_lC_mMeFen3h100n3pI_tBb100e3lCle1E-05oAol0.001_s0/Confusion_matrix.png)

#### Sweep over parameters

```
juml sweep -h

juml sweep --model LinearModel --dataset LinearDataset --dataset.LinearDataset.input_dim 5 --dataset.LinearDataset.output_dim 10 --print_level 1 --Sweeper.seeds 1 2 3 --Sweeper.params '{"trainer.BpSp.epochs":[100,200,300],"trainer.BpSp.optimiser.Adam.lr":[1e-5,1e-4,1e-3,1e-2]}' --Sweeper.log_x trainer.BpSp.optimiser.Adam.lr --Sweeper.devices "[[],[],[],[],[],[]]" --Sweeper.no_cache
```

[`[ sweep_results ]`](results/sweep/dLi5o10te200tr200ts0.0x0.0lMmLeIpItBb100e1,2,300lCle1E-05oAol,1E-0.0001,.001,.01,5s1,2,3/results.md)

### Extending `juml`

*TODO*

## Citation

If you find JUML helpful in your research, please cite:

```
@misc{levi_juml_2025,
	title = {{JUML}: {A} {Judicious}, {Unified} and extendable framework for multi-paradigm {Machine} {Learning} research},
	shorttitle = {{JUML}},
	url = {https://github.com/jakelevi1996/juml},
	abstract = {A Judicious, Unified and extendable framework for multi-paradigm Machine Learning research, powered by jutility and PyTorch.},
	author = {Levi, Jake},
	year = {2025},
}
```
