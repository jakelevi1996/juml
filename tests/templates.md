# Unit test templates

See [`tests/templates.md` on GitHub](https://github.com/jakelevi1996/juml/blob/main/tests/templates.md).

## Contents

- [Unit test templates](#unit-test-templates)
  - [Contents](#contents)
  - [Models](#models)
    - [Embed](#embed)
    - [Pool](#pool)
  - [Datasets](#datasets)
  - [Loss](#loss)
  - [Trainers](#trainers)
  - [Commands](#commands)

## Models

```py
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_<model_type>")

def test_<model_type>():
    printer = util.Printer("test_<model_type>", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_<model_type>")

    input_dim   = <input_dim>
    output_dim  = <output_dim>
    x = torch.rand([<input_shape>, <input_dim>])
    t = torch.rand([<output_shape>, <output_dim>])

    model = juml.models.<ModelType>(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        ...,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss        = juml.loss.<LossType>()
    optimiser   = torch.optim.Adam(model.parameters())

    assert repr(model) == "<ModelType>(num_params=<num_params_str>)"
    assert model.num_params() == <num_params>

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert y_0.dtype is torch.float32
    assert y_0.dtype is not torch.int64
    assert list(y_0.shape) == <model_output_shape>

    loss_0 = loss.forward(y_0, t)
    loss_0.backward()
    optimiser.step()

    y_1 = model.forward(x)
    loss_1 = loss.forward(y_1, t)

    assert loss_1.item() < loss_0.item()
```

### Embed

### Pool

## Datasets

```py
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_datasets/test_<dataset_type>")

def test_<dataset_type>():
    printer = util.Printer("test_<dataset_type>", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_<dataset_type>")

    dataset = juml.datasets.<DatasetType>(
        ...,
    )
    assert repr(dataset) == "<DatasetType>(n_train=<n_train_str>, n_test=<n_test_str>)"
    assert dataset.get_input_shape()    == [<input_shape>]
    assert dataset.get_output_shape()   == [<output_shape>]

    train_split = dataset.get_data_split("train")
    assert isinstance(train_split, torch.utils.data.Dataset)
    assert len(train_split) == <n_train>

    test_split = dataset.get_data_split("test")
    assert isinstance(test_split, torch.utils.data.Dataset)
    assert len(test_split) == <n_test>

    batch_size = 83
    data_loader = dataset.get_data_loader("train", batch_size)
    x, t = next(iter(data_loader))

    assert isinstance(x, torch.Tensor)
    assert list(x.shape) == [batch_size, <x_shape>]
    assert x.dtype is torch.float32
    assert x.dtype is not torch.int64
    assert x.max().item() <= <x_max>
    assert x.min().item() >= <x_min>

    assert isinstance(t, torch.Tensor)
    assert list(t.shape) == [batch_size, <t_shape>]
    assert t.dtype is torch.<output_type>
    assert t.dtype is not torch.<not_output_type>
    assert t.max().item() <= <t_max>
    assert t.min().item() >= <t_min>
```

## Loss

## Trainers

## Commands
