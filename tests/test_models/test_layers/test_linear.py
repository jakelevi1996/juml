import torch
import juml

def test_linearlayer_normalise():
    juml.util.set_torch_seed("test_linearlayer_normalise")

    input_dim = 13
    output_dim = 7
    batch_size = 64
    x_ni = torch.normal(3.4, 105.6, [batch_size, input_dim])

    layer = juml.models.layers.LinearLayer(input_dim, output_dim)

    y_no = layer.forward(x_ni)
    assert y_no.mean(0).abs().max().item() > 10
    assert (y_no.std(0) - 1).abs().max().item() > 10
    assert list(y_no.shape) == [batch_size, output_dim]

    layer.normalise(x_ni)

    y_no = layer.forward(x_ni)
    assert y_no.mean(0).abs().max().item() > 0.1
    assert y_no.mean(0).abs().max().item() < 10.0
    assert (y_no.std(0) - 1).abs().max().item() < 1e-5
    assert list(y_no.shape) == [batch_size, output_dim]

    layer.normalise(x_ni, 0.0)

    y_no = layer.forward(x_ni)
    assert y_no.mean(0).abs().max().item() < 1e-5
    assert (y_no.std(0) - 1).abs().max().item() < 1e-5
    assert list(y_no.shape) == [batch_size, output_dim]
