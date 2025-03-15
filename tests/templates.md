# Unit test templates

See [link](URL).

## Models

```py
import torch
from jutility import util
import juml

OUTPUT_DIR = juml.test_utils.get_output_dir("test_models/test_<model_type>")

def test_<model_type>():
    printer = util.Printer("test_<model_type>", dir_name=OUTPUT_DIR)
    juml.test_utils.set_torch_seed("test_<model_type>")

    input_dim   = 7
    output_dim  = 11
    x = torch.rand([3, 4, 5, input_dim])
    t = torch.rand([3, 6, 7, 8, output_dim])

    model = juml.models.<ModelType>(
        input_shape=list(x.shape),
        output_shape=list(t.shape),
        ...,
        embedder=juml.models.embed.Identity(),
        pooler=juml.models.pool.Identity(),
    )
    loss        = juml.loss.<LossType>()
    optimiser   = torch.optim.Adam(model.parameters())

    assert repr(model) == "<ModelType>(num_params=1234)"
    assert model.num_params() == 1234

    y_0 = model.forward(x)
    assert isinstance(y_0, torch.Tensor)
    assert y_0.dtype is torch.float32
    assert y_0.dtype is not torch.int64
    assert list(y_0.shape) == [9, 10, output_dim]

    loss_0 = loss.forward(y_0, t)
    loss_0.backward()
    optimiser.step()

    y_1 = model.forward(x)
    loss_1 = loss.forward(y_1, t)

    assert loss_1.item() < loss_0.item()
```
