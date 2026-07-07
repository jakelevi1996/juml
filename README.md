# JUML

A Judicious, Unified, extendable, paradigm-agnostic framework for Machine Learning research, powered by [`jutility`](https://github.com/jakelevi1996/jutility) and [PyTorch](https://pytorch.org/).

> *[Judicious [adjective]: having or showing reason and good judgment in making decisions](https://dictionary.cambridge.org/dictionary/english/judicious)*

![](https://github.com/jakelevi1996/juml/raw/main/img/logo_black.png)

## Contents

- [JUML](#juml)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Overview](#overview)
  - [Usage examples](#usage-examples)
  - [Extension guide](#extension-guide)
  - [Extra tips](#extra-tips)
  - [Citation](#citation)

## Installation

JUML is available as [a Python package on PyPI](https://pypi.org/project/juml-framework/), and can be installed using `pip` with the following commands:

```
python -m pip install -U pip
python -m pip install -U juml-framework
```

Alternatively, JUML can be installed in "editable mode" from the [GitHub repository](https://github.com/jakelevi1996/juml):

```
git clone https://github.com/jakelevi1996/juml.git
python -m pip install -U pip
python -m pip install -e ./juml
```

JUML depends on [PyTorch](https://pytorch.org/). The installation instructions for PyTorch depend on which (if any) CUDA version is available, so PyTorch won't be automatically installed by `pip` when installing JUML. Instead, please install PyTorch following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Overview

The purpose of JUML is to make the process of running ML experiments smoother.

For example, as described in more detail in the [Extension guide](#extension-guide), when starting a new project, all you have to do is:

- Appropriately define one or more training loops, models, and datasets (or RL environments)
- Define subclasses of [`Framework`](https://github.com/jakelevi1996/juml/blob/main/src/juml/framework.py) and [`Sweep`](https://github.com/jakelevi1996/juml/blob/main/src/juml/commands/sweep.py), and override one method in each of those subclasses
- Configure your `pyproject.toml` file appropriately

Then, JUML will automatically provide you with:

- A CLI, including configurable selection and automatic nested initialisation of objects and subcommands (without having to write any config files)
- Automatic results directory naming, so you don't have to worry about results overwriting each other, or manually naming each results file, or writing code to do this yourself, or paying a subscription for an external tool
- Automatic sweeping over multiple parameters simultaneously and plotting sweep results *from the command line* (without having to write any extra sweeping code)
- (and possibly more features to be added in future)

## Usage examples

Once you install JUML, you can run a few commands and experiments from the command-line, without having to write any code.

First of all, view the top-level and command-specific help interfaces:

```
juml -h
juml TrainClassification -h
juml Sweep -h
```

Train an MLP on XOR to 100% accuracy in under 1 second (CPU):

```
juml TrainClassification \
    --dataset Xor \
    --model ReluMlp \
    --model.ReluMlp.hidden_dim 10 \
    --epochs 200
```

```
cli: Xor()
cli: ReluMlp(depth=2, hidden_dim=10, input_dim=2, output_dim=2)
cli: DeviceConfig(gpu=False, visible_devices=[])
Count | Time        | Epoch      | Batch      | Loss       | Train acc  | Test acc
----- | ----------- | ---------- | ---------- | ---------- | ---------- | ----------
0     | 0.0010s     |          0 |          0 |    0.72485 |            |
0     | 0.0010s     |          0 |          0 |    0.72485 |            |
1     | 0.0011s     |          0 |            |            |    0.25000 |    0.25000
...
398   | 0.0804s     |        199 |          0 |    0.55982 |            |
399   | 0.0805s     |        199 |            |            |    1.00000 |    1.00000
Saving in "results/trainclassification/b100dXe200mRmd2mh10s0/args.json"
Saving in "results/trainclassification/b100dXe200mRmd2mh10s0/cmd.txt"
Saving in "results/trainclassification/b100dXe200mRmd2mh10s0/table.pkl"
Saving in "results/trainclassification/b100dXe200mRmd2mh10s0/metrics.json"
Saving in "results/trainclassification/b100dXe200mRmd2mh10s0/metrics.png"
Time taken for `TrainClassification` = 0.5844 seconds
```

![](https://github.com/jakelevi1996/juml/raw/main/results/trainclassification/b100dXe200mRmd2mh10s0/metrics.png)

Train an MLP on MNIST to 98% test accuracy in under 30 seconds (CPU):

```
juml TrainClassification \
    --dataset Mnist \
    --model ReluMlp \
    --model.ReluMlp.hidden_dim 1000 \
    --epochs 5
```

```
cli: Mnist(flat=True)
cli: ReluMlp(depth=2, hidden_dim=1000, input_shape=[784], output_shape=[10])
Count | Time        | Epoch      | Batch      | Loss       | Train acc  | Test acc
----- | ----------- | ---------- | ---------- | ---------- | ---------- | ----------
0     | 0.0142s     |          0 |          0 |    2.35821 |            |
91    | 1.0022s     |          0 |         91 |    0.21197 |            |
183   | 2.0046s     |          0 |        183 |    0.18111 |            |
273   | 3.0019s     |          0 |        273 |    0.18510 |            |
364   | 4.0017s     |          0 |        364 |    0.14397 |            |
456   | 5.0047s     |          0 |        456 |    0.14908 |            |
547   | 6.0080s     |          0 |        547 |    0.11449 |            |
599   | 6.5894s     |          0 |        599 |    0.07091 |            |
600   | 6.9580s     |          0 |            |            |    0.97190 |    0.96670
604   | 7.0040s     |          1 |          3 |    0.09275 |            |
694   | 8.0038s     |          1 |         93 |    0.07405 |            |
812   | 9.0012s     |          1 |        211 |    0.10806 |            |
983   | 10.0041s    |          1 |        382 |    0.08695 |            |
1150  | 11.0005s    |          1 |        549 |    0.08830 |            |
1200  | 11.2970s    |          1 |        599 |    0.03669 |            |
1201  | 11.6915s    |          1 |            |            |    0.98123 |    0.97070
1253  | 12.0027s    |          2 |         51 |    0.05098 |            |
1418  | 13.0049s    |          2 |        216 |    0.05247 |            |
1578  | 14.0035s    |          2 |        376 |    0.08137 |            |
1739  | 15.0028s    |          2 |        537 |    0.08548 |            |
1801  | 15.3885s    |          2 |        599 |    0.05897 |            |
1802  | 15.7840s    |          2 |            |            |    0.99040 |    0.97830
1838  | 16.0027s    |          3 |         35 |    0.04630 |            |
1996  | 17.0021s    |          3 |        193 |    0.12142 |            |
2148  | 18.0031s    |          3 |        345 |    0.01137 |            |
2300  | 19.0052s    |          3 |        497 |    0.01614 |            |
2402  | 19.6594s    |          3 |        599 |    0.04586 |            |
2403  | 20.0639s    |          3 |            |            |    0.99268 |    0.97970
2404  | 20.0704s    |          4 |          0 |    0.00618 |            |
2550  | 21.0057s    |          4 |        146 |    0.07377 |            |
2706  | 22.0042s    |          4 |        302 |    0.03206 |            |
2858  | 23.0003s    |          4 |        454 |    0.02141 |            |
3003  | 23.9617s    |          4 |        599 |    0.00631 |            |
3004  | 24.3791s    |          4 |            |            |    0.99547 |    0.98180
Saving in "results/trainclassification/b100dMdfTe5mRmd2mh1000s0/args.json"
Saving in "results/trainclassification/b100dMdfTe5mRmd2mh1000s0/cmd.txt"
Saving in "results/trainclassification/b100dMdfTe5mRmd2mh1000s0/metrics.json"
Saving in "results/trainclassification/b100dMdfTe5mRmd2mh1000s0/metrics.png"
Time taken for `TrainClassification` = 26.7501 seconds
```

![](https://github.com/jakelevi1996/juml/raw/main/results/trainclassification/b100dMdfTe5mRmd2mh1000s0/metrics.png)

Sweep over width, depth, and random seeds for the MLP:

```
juml Sweep \
    --params '{
        "seed": [0, 1, 2],
        "model.ReluMlp.hidden_dim": [20, 50, 100, 200, 500, 1000],
        "model.ReluMlp.depth": [1, 2, 3]
    }' \
    --PlottingConfig.target_metric final_test_acc \
    --PlottingConfig.x_key model.ReluMlp.hidden_dim \
    --PlottingConfig.c_key model.ReluMlp.depth \
    --PlottingConfig.log_x \
    TrainClassification \
    --dataset Mnist \
    --model ReluMlp \
    --epochs 3
```

This trains 54 models in ~12 minutes, and produces the following graph:

![](https://github.com/jakelevi1996/juml/raw/main/results/sweep/b100dMdfTe3mRmd1,2,3mh1,2,50,0,00s0,1,2/sweep_results.png)

Tidy up the labels and colour scheme by using the appropriate `--PlottingConfig` arguments. We also provide the `--name` argument to the `Sweep` command, because although `Sweep` automatically names output directories, it will overwrite our previous graph if we only change `--PlottingConfig` arguments. The results will automatically be loaded from disk without rerunning the experiments unless we specify `Sweep --force_run`:

```
juml Sweep \
    --params '{
        "seed": [0, 1, 2],
        "model.ReluMlp.hidden_dim": [20, 50, 100, 200, 500, 1000],
        "model.ReluMlp.depth": [1, 2, 3]
    }' \
    --name tidy_mnist_sweep \
    --PlottingConfig.target_metric final_test_acc \
    --PlottingConfig.x_key model.ReluMlp.hidden_dim \
    --PlottingConfig.c_key model.ReluMlp.depth \
    --PlottingConfig.log_x \
    --PlottingConfig.x_label Width \
    --PlottingConfig.y_label 'Test acc' \
    --PlottingConfig.c_label Depth \
    --PlottingConfig.ylim 0.9 1 \
    --PlottingConfig.cool_colours \
    TrainClassification \
    --dataset Mnist \
    --model ReluMlp \
    --epochs 3
```

![](https://github.com/jakelevi1996/juml/raw/main/results/sweep/tidy_mnist_sweep/sweep_results.png)

We can transpose the axes of the graph simply by swapping `--PlottingConfig` arguments and choosing a new name:

```
juml Sweep \
    --params '{
        "seed": [0, 1, 2],
        "model.ReluMlp.hidden_dim": [20, 50, 100, 200, 500, 1000],
        "model.ReluMlp.depth": [1, 2, 3]
    }' \
    --name tidy_mnist_sweep_transpose \
    --PlottingConfig.target_metric final_test_acc \
    --PlottingConfig.x_key model.ReluMlp.depth \
    --PlottingConfig.c_key model.ReluMlp.hidden_dim \
    --PlottingConfig.x_label Depth \
    --PlottingConfig.y_label 'Test acc' \
    --PlottingConfig.c_label Width \
    --PlottingConfig.ylim 0.9 1 \
    --PlottingConfig.cool_colours \
    TrainClassification \
    --dataset Mnist \
    --model ReluMlp \
    --epochs 3
```

![](https://github.com/jakelevi1996/juml/raw/main/results/sweep/tidy_mnist_sweep_transpose/sweep_results.png)

## Extension guide

JUML is designed to be both extendable and flexible. Here we provide a guide for extending JUML in your project, step by step. You don't have to do everything in this guide, and you can do more than we outline here.

We will assume that your project is called `MYPROJ`, but you should replace `MYPROJ` with a short and descriptive (and lowercase) name for your project. See the [JUML source code](https://github.com/jakelevi1996/juml/tree/main/src/juml) for recommended file and directory structure.

Here are the steps:

1. Define models:
   1. Define an interface class for your models, which is a subclass of [`juml.models.Model`](https://github.com/jakelevi1996/juml/blob/main/src/juml/models/model.py) (see [`juml.models.FeedForwardModel`](https://github.com/jakelevi1996/juml/blob/main/src/juml/models/ff.py) for an example)
   2. Define some model classes which are subclasses of your model interface class (see [`juml.models.ReluMlp`](https://github.com/jakelevi1996/juml/blob/main/src/juml/models/relu_mlp.py) for an example)
   3. Define a function `get_all_models` which returns a list of your defined model classes (see [`juml.models.get_all_models`](https://github.com/jakelevi1996/juml/blob/main/src/juml/models/__init__.py) for an example)
2. Define datasets:
   1. Define an interface class for your datasets, which is a subclass of [`juml.data.Dataset`](https://github.com/jakelevi1996/juml/blob/main/src/juml/data/dataset.py) (see [`juml.data.ClassificationDataset`](https://github.com/jakelevi1996/juml/blob/main/src/juml/data/classification.py) for an example)
   2. Define some dataset classes which are subclasses of your dataset interface class (see [`juml.data.Mnist`](https://github.com/jakelevi1996/juml/blob/main/src/juml/data/mnist.py) for an example)
   3. Define a function `get_all_datasets` which returns a list of your defined dataset classes (see [`juml.data.get_all_datasets`](https://github.com/jakelevi1996/juml/blob/main/src/juml/data/__init__.py) for an example)
3. Define commands:
   1. Define some training loops or other commands, which are subclasses of [`juml.commands.Command`](https://github.com/jakelevi1996/juml/blob/main/src/juml/commands/command.py) (see [`juml.commands.TrainClassification`](https://github.com/jakelevi1996/juml/blob/main/src/juml/commands/train_classification.py) for an example)
   2. Define a class `Sweep` which is a subclass of [`juml.commands.Sweep`](https://github.com/jakelevi1996/juml/blob/main/src/juml/commands/sweep.py), override the class-method `get_subcommands`, and from it return a list of your defined command classes (NOT including your `Sweep` subclass itself)
   3. Define a function `get_all_commands` which returns a list of your defined command classes, including your `Sweep` subclass (see [`juml.commands.get_all_commands`](https://github.com/jakelevi1996/juml/blob/main/src/juml/commands/__init__.py) for an example)
4. Define framework, configure and install project:
   1. Define a class `Framework` in the file `src/MYPROJ/framework.py` which is a subclass of [`juml.Framework`](https://github.com/jakelevi1996/juml/blob/main/src/juml/framework.py), override the class-method `get_commands`, and from it return the output from your `get_all_commands` function
   2. Make a file `pyproject.toml` following the template below
   3. Run the commands `python -m pip install -U pip` and `python -m pip install -e .`

`pyproject.toml` template:

```
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "MYPROJ"
version = "0.0.1"

[project.scripts]
MYPROJ = "MYPROJ.framework:Framework.run"
```

Now you are ready to run some commands and sweeps! For example:

```
MYPROJ MyCommand ...
MYPROJ Sweep --params '{...}' ... MyCommand ...`
```

## Extra tips

There are different approaches that can be taken to debugging. Perhaps the easiest is to make a new Python script which calls `juml.Framework.run` (or your subclass of `Framework`) directly with specified arguments, and then debug this like a normal script, for example:

```py
import juml

s = (
    "TrainClassification "
    "--dataset Mnist "
    "--model ReluMlp "
    "--model.ReluMlp.hidden_dim 1000 "
    "--epochs 1"
)
juml.Framework.run(s.split())
```

Sometimes you may want to make more sophisticated plots beyond those produced by `juml Sweep` (or your subclass of `Sweep`), which involves iterating over arguments and loading saved metrics. Every subclass of [`juml.commands.Command`](https://github.com/jakelevi1996/juml/blob/main/src/juml/commands/command.py) inherits a class method `load_metric_from_args`, to which you can simply provide a dictionary of arguments (for example those saved in [`args.json`](https://github.com/jakelevi1996/juml/blob/main/results/trainclassification/b100dMdfTe5mRmd2mh1000s0/args.json) by your training command) and the name of the metric you want to load. Then `load_metric_from_args` will load that metric for you. For example:

```py
m = juml.commands.TrainClassification.load_metric_from_args(
    arg_dict={
        "seed": 0,
        "epochs": 5,
        "batch_size": 100,
        "dataset": "Mnist",
        "model": "ReluMlp",
        "model.ReluMlp.depth": 2,
        "model.ReluMlp.hidden_dim": 1000
    },
    name="final_test_acc",
)
print(m)
# >>> 0.9818000197410583
```

## Citation

If you find JUML helpful in your research, please cite:

```
@misc{levi_juml_2025,
    title = {{JUML}: {A} {Judicious}, {Unified}, extendable, paradigm-agnostic framework for {Machine} {Learning} research},
    shorttitle = {{JUML}},
    url = {https://github.com/jakelevi1996/juml},
    abstract = {A Judicious, Unified, extendable, paradigm-agnostic framework for Machine Learning research, powered by jutility and PyTorch.},
    author = {Levi, Jake},
    year = {2025},
}
```
