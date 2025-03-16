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
    - [Extending `juml`](#extending-juml)

## Installation

The `juml` package is available as [a Python package on PyPI](https://pypi.org/project/juml-toolkit/), and can be installed using `pip` with the following commands:

```
python -m pip install -U pip
python -m pip install -U juml-toolkit
```

Alternatively, `juml` can be installed in "editable mode" from the GitHub repository:

```
git clone https://github.com/jakelevi1996/juml.git
python -m pip install -U pip
python -m pip install -e ./juml
```

## Overview

The `juml` framework defines 6 fundamental classes (and several example subclasses), available in the `juml.base` namespace module, which are expected to be subclassed in downstream projects:

- [`juml.base.Model`](src/juml/models/base.py)
- [`juml.base.Dataset`](src/juml/datasets/base.py)
- [`juml.base.Loss`](src/juml/loss/base.py)
- [`juml.base.Trainer`](src/juml/train/base.py)
- [`juml.base.Command`](src/juml/commands/base.py)
- [`juml.base.Framework`](src/juml/framework.py)

*Coming soon: `juml.base.Environment` for RL*

## Usage examples

*TODO*

### Out of the box

*TODO*

### Extending `juml`

*TODO*
