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

def test_linearlayer_lstsq():
    juml.util.set_torch_seed("test_linearlayer_lstsq")

    input_dim = 13
    output_dim = 7
    batch_size = 64
    x_ni = torch.normal(3.4, 105.6, [batch_size, input_dim])
    w_io = torch.normal(0, 1, [input_dim, output_dim])
    b_o = torch.normal(0, 1, [output_dim])
    t_no = x_ni @ w_io + b_o

    layer = juml.models.layers.LinearLayer(input_dim, output_dim)

    y_no = layer.forward(x_ni)
    assert (y_no - t_no).square().mean().item() > 100.0
    assert list(y_no.shape) == [batch_size, output_dim]

    layer.normalise(x_ni)

    y_no = layer.forward(x_ni)
    assert (y_no - t_no).square().mean().item() > 100.0
    assert list(y_no.shape) == [batch_size, output_dim]

    layer.lstsq(x_ni, t_no, 1e-3)

    y_no = layer.forward(x_ni)
    assert (y_no - t_no).square().mean().item() < 1e-5
    assert list(y_no.shape) == [batch_size, output_dim]

def test_linearlayer_indirect_lstsq():
    juml.util.set_torch_seed("test_linearlayer_indirect_lstsq")

    input_dim = 13
    output_dim = 7
    residual_dim = 11

    batch_size = 64
    x_ni = torch.normal(3.4, 105.6, [batch_size, input_dim])
    w_io = torch.normal(0, 1, [input_dim, residual_dim])
    b_o  = torch.normal(0, 1, [residual_dim])
    a_od = torch.normal(0, 1, [residual_dim, output_dim])
    d_nd  = torch.normal(0, 1, [batch_size, output_dim])
    t_nd = (x_ni @ w_io + b_o) @ a_od + d_nd


    layer = juml.models.layers.LinearLayer(input_dim, residual_dim)

    y_nd = layer.forward(x_ni) @ a_od + d_nd
    assert (y_nd - t_nd).square().mean().item() > 100
    assert list(y_nd.shape) == [batch_size, output_dim]

    layer.indirect_lstsq(x_ni, t_nd, a_od, d_nd, 1e-7, 1e-1)

    y_nd = layer.forward(x_ni) @ a_od + d_nd
    assert (y_nd - t_nd).square().mean().item() < 1e-5
    assert list(y_nd.shape) == [batch_size, output_dim]
