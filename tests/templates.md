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

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_{<modeltype>}")

def test_{<modeltype>}():
    printer = util.Printer("test_{<modeltype>}", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<modeltype>}")

    x = torch.rand([{<x_shape>}])
    t = torch.rand([{<t_shape>}])

    model = juml.models.{<ModelType>}(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        ...,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.{<LossType>}()
    optimiser = torch.optim.Adam(model.parameters())

    assert repr(model) == "{<ModelType>}(num_params={<num_params_str>})"
    assert model.num_params() == {<num_params>}

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert y_0.dtype is torch.float32
    assert y_0.dtype is not torch.int64
    assert list(y_0.shape) == list(t.shape)
    printer(y_0.max(), y_0.min())
    assert y_0.max().item() <= {<y_0_max>}
    assert y_0.min().item() >= -{<y_0_max>}

    loss_0 = loss.forward(y_0, t)
    loss_0.backward()
    optimiser.step()

    y_1 = model.forward(x)
    loss_1 = loss.forward(y_1, t)

    printer(loss_0, loss_1)
    assert loss_1.item() < loss_0.item()

    printer(repr(list(model.layers)))
    assert repr(list(model.layers)) == (
        "[{<repr_layers>}]"
    )
```

### Embed

```py
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_embed")

def test_{<embeddertype>}():
    printer = util.Printer("test_{<embeddertype>}", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<embeddertype>}")

    embedder = juml.models.embed.{<EmbedderType>}(...)

    x = torch.rand({<x_shape>})

    embedder.set_input_shape(list(x.shape))
    assert repr(embedder) == "{<EmbedderType>}(num_params={<num_params_str>})"
    assert embedder.num_params() == {<num_params>}
    assert embedder.get_output_shape() == [{<output_shape>}]
    assert embedder.get_output_dim(-1) == {<output_dim>}

    y = embedder.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == {<y_shape>}
    printer(y.max(), y.min())
    assert y.max().item() <= {<y_max>}
    assert y.min().item() >= -{<y_max>}

def test_{<embeddertype>}_model():
    printer = util.Printer("test_{<embeddertype>}_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<embeddertype>}_model")

    embedder = juml.models.embed.{<EmbedderType>}(...)

    x = torch.rand([{<x_shape>}])
    t = torch.rand([{<t_shape>}])

    model = juml.models.{<ModelType>}(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        ...,
        embedder=embedder,
        pooler=juml.models.pool.Identity(),
    )
    loss = juml.loss.{<LossType>}()
    optimiser = torch.optim.Adam(model.parameters())
    assert model.embed is embedder

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert y_0.dtype is torch.float32
    assert y_0.dtype is not torch.int64
    assert list(y_0.shape) == list(t.shape)
    printer(y_0.max(), y_0.min())
    assert y_0.max().item() <= {<y_0_max>}
    assert y_0.min().item() >= -{<y_0_max>}

    loss_0 = loss.forward(y_0, t)
    loss_0.backward()
    optimiser.step()

    y_1 = model.forward(x)
    loss_1 = loss.forward(y_1, t)

    printer(loss_0, loss_1)
    assert loss_1.item() < loss_0.item()
```

### Pool

```py
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_pool")

def test_{<pooltype>}():
    printer = util.Printer("test_{<pooltype>}", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<pooltype>}")

    pooler = juml.models.pool.{<PoolType>}(...)

    x = torch.rand([{<x_shape>}])

    pooler.set_shapes(list(x.shape), [{<output_shape>}])
    assert repr(pooler) == "{<PoolType>}(num_params={<num_params_str>})"
    assert pooler.num_params() == {<num_params>}
    assert pooler.get_input_shape() == [{<input_shape>}]
    assert pooler.get_input_dim(-1) == {<input_dim>}

    y = pooler.forward(x)
    assert isinstance(y, torch.Tensor)
    assert y.dtype is torch.float32
    assert y.dtype is not torch.int64
    assert list(y.shape) == {<y_shape>}
    printer(y.max(), y.min())
    assert y.max().item() <= {<y_max>}
    assert y.min().item() >= -{<y_max>}

def test_{<pooltype>}_model():
    printer = util.Printer("test_{<pooltype>}_model", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<pooltype>}_model")

    pooler = juml.models.pool.{<PoolType>}(...)

    x = torch.rand([{<x_shape>}])
    t = torch.rand([{<t_shape>}])

    model = juml.models.{<ModelType>}(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        ...,
        embedder=juml.models.embed.Identity(),
        pooler=pooler,
    )
    loss = juml.loss.{<LossType>}()
    optimiser = torch.optim.Adam(model.parameters())
    assert model.pool is pooler

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert y_0.dtype is torch.float32
    assert y_0.dtype is not torch.int64
    assert list(y_0.shape) == list(t.shape)
    printer(y_0.max(), y_0.min())
    assert y_0.max().item() <= {<y_0_max>}
    assert y_0.min().item() >= -{<y_0_max>}

    loss_0 = loss.forward(y_0, t)
    loss_0.backward()
    optimiser.step()

    y_1 = model.forward(x)
    loss_1 = loss.forward(y_1, t)

    printer(loss_0, loss_1)
    assert loss_1.item() < loss_0.item()
```

## Datasets

```py
import pytest
import torch
import torch.utils.data
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_datasets/test_{<datasettype>}")

def test_{<datasettype>}():
    printer = util.Printer("test_{<datasettype>}", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<datasettype>}")

    dataset = juml.datasets.{<DatasetType>}(
        ...,
    )
    assert repr(dataset) == "{<DatasetType>}(n_train={<n_train_str>}, n_test={<n_test_str>})"
    assert dataset.get_input_shape()    == [{<input_shape>}]
    assert dataset.get_output_shape()   == [{<output_shape>}]
    assert dataset.get_default_loss()   is juml.loss.{<LossType>}

    train_split = dataset.get_data_split("train")
    assert isinstance(train_split, torch.utils.data.Dataset)
    assert len(train_split) == {<n_train>}

    test_split = dataset.get_data_split("test")
    assert isinstance(test_split, torch.utils.data.Dataset)
    assert len(test_split) == {<n_test>}

    batch_size = {<batch_size>}
    data_loader = dataset.get_data_loader("train", batch_size)
    x, t = next(iter(data_loader))

    assert isinstance(x, torch.Tensor)
    assert x.dtype is torch.float32
    assert x.dtype is not torch.int64
    assert list(x.shape) == [batch_size, {<x_shape>}]
    printer(x.max(), x.min())
    assert x.max().item() <= {<x_max>}
    assert x.min().item() >= -{<x_max>}

    assert isinstance(t, torch.Tensor)
    assert t.dtype is torch.float32
    assert t.dtype is not torch.int64
    assert list(t.shape) == [batch_size, {<t_shape>}]
    printer(t.max(), t.min())
    assert t.max().item() <= {<t_max>}
    assert t.min().item() >= -{<t_max>}

    printer(dataset.get_loss_weights())
    assert isinstance(dataset.get_loss_weights(), torch.Tensor)
    assert list(dataset.get_loss_weights().shape) == [output_dim]
    # OR
    with pytest.raises(NotImplementedError):
        dataset.get_loss_weights()
```

## Loss

```py
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_loss/test_{<losstype>}")

def test_{<losstype>}():
    printer = util.Printer("test_{<losstype>}", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<losstype>}")

    batch_size = {<batch_size>}
    output_dim = {<output_dim>}
    y = torch.rand([batch_size, output_dim])
    t = torch.rand([batch_size, output_dim])
    # OR
    batch_size  = {<batch_size>}
    num_classes = {<num_classes>}
    y = torch.rand([batch_size, num_classes])
    t = torch.randint(0, num_classes, [batch_size])

    loss = juml.batch_loss.{<LossType>}()

    batch_loss = loss.forward(y, t)
    assert isinstance(batch_loss, torch.Tensor)
    assert batch_loss.dtype is torch.float32
    assert batch_loss.dtype is not torch.int64
    assert list(batch_loss.shape) == []
    printer(batch_loss)
    assert batch_loss.item() >= 0
    assert batch_loss.item() <= {<max_batch_loss>}

    metric = loss.metric_batch(y, t)
    printer(metric)
    assert isinstance(metric, float)
    # OR
    assert isinstance(metric, int)
    assert metric >= 0
    assert metric <= {<max_metric>}

def test_{<losstype>}_cli():
    printer = util.Printer("test_{<losstype>}_cli", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<losstype>}_cli")

    parser  = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--loss {<LossType>} "
        "--dataset {<DatasetType>} "
        ...
        "--model {<ModelType>} "
        ...
        "--trainer {<TrainerType>} "
        ...
    )
    args = parser.parse_args(args_str.split())
    trainer = args.get_command().run(args)
    assert isinstance(trainer, juml.train.{<TrainerType>})

    loss = trainer.loss
    assert isinstance(loss, juml.loss.{<LossType>})
    assert loss.weights is None
    assert not isinstance(loss.weights, torch.Tensor)
    # OR
    assert isinstance(loss.weights, torch.Tensor)
    assert loss.weights is not None

    batch_size = {<batch_size>}
    optimiser = torch.optim.Adam(trainer.model.parameters())
    x, t = next(iter(trainer.dataset.get_data_loader("train", batch_size)))
    y_0 = trainer.model.forward(x)
    loss_0 = loss.forward(y_0, t)
    assert isinstance(x,        torch.Tensor)
    assert isinstance(t,        torch.Tensor)
    assert isinstance(y_0,      torch.Tensor)
    assert isinstance(loss_0,   torch.Tensor)
    assert list(x.shape         ) == [{<x_shape>}]
    assert list(t.shape         ) == [{<t_shape>}]
    assert list(y_0.shape       ) == [{<y_0_shape>}]
    assert list(loss_0.shape    ) == [{<loss_0_shape>}]

    optimiser.zero_grad()
    loss_0.backward()
    optimiser.step()

    y_1 = trainer.model.forward(x)
    loss_1 = loss.forward(y_1, t)

    printer(loss_0, loss_1)
    assert loss_1.item() < loss_0.item()
```

## Trainers

```py
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_train/test_{<trainertype>}")

def test_{<trainertype>}():
    printer = util.Printer("test_{<trainertype>}", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<trainertype>}")

    parser  = juml.base.Framework.get_parser()
    args_str = (
        "train "
        "--trainer {<TrainerType>} "
        ...
        "--model {<ModelType>} "
        ...
        "--dataset {<DatasetType>} "
        ...
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.Train)

    trainer = command.run(args)
    assert isinstance(trainer,          juml.train.{<TrainerType>})
    assert isinstance(trainer.model,    juml.models.{<ModelType>})
    assert isinstance(trainer.dataset,  juml.datasets.{<DatasetType>})
    assert isinstance(trainer.loss,     juml.loss.{<LossType>})
    assert isinstance(trainer.table,    util.Table)

    batch_loss = trainer.table.get_data("batch_loss")
    printer(batch_loss)
    assert batch_loss[-1] < batch_loss[0]

    batch_size = {<batch_size>}
    x, t = next(iter(trainer.dataset.get_data_loader("train", batch_size)))
    y = trainer.model.forward(x)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    assert list(x.shape) == [{<x_shape>}]
    assert list(y.shape) == [{<y_shape>}]
    assert list(t.shape) == [{<t_shape>}]
```

## Commands

```py
import os
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_commands/test_{<commandtype>}")

def test_{<commandtype>}():
    printer = util.Printer("test_{<commandtype>}", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_{<commandtype>}")

    output_path = (
        "results/{<output_path>}"
    )
    if os.path.isfile(output_path):
        os.remove(output_path)

    assert not os.path.isfile(output_path)

    parser = juml.base.Framework.get_parser()
    args_str = (
        "{<CommandType>} "
        ...
    )
    args = parser.parse_args(args_str.split())
    command = args.get_command()
    assert isinstance(command, juml.commands.{<CommandType>})

    {<return_value>} = command.run(args)
    assert isinstance({<return_value>}, juml.{<ReturnType>})

    assert os.path.isfile(output_path)
```
