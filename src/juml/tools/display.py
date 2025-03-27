import torch
from jutility import plotting, util, units
from juml.models.sequential import Sequential

def display_sequential(
    model:      Sequential,
    x:          torch.Tensor,
    printer:    (util.Printer | None)=None,
) -> tuple[torch.Tensor, util.Table]:
    if printer is None:
        printer = util.Printer()

    printer.hline()
    table = util.Table(
        util.Column("layer",    "r",    -40),
        util.Column("shape",    "s",    -22),
        util.Column("time",     ".5fs", 11),
    )
    time_list   = []
    total_timer = util.Timer(verbose=False)
    layer_timer = util.Timer(verbose=False)
    display_layer("Input", x, printer, layer_timer, [])

    with layer_timer:
        x = model.embed.forward(x)
        display_layer(model.embed, x, printer, layer_timer, time_list)
    for layer in model.layers:
        with layer_timer:
            x = layer.forward(x)
            display_layer(layer, x, printer, layer_timer, time_list)
    with layer_timer:
        x = model.pool.forward(x)
        display_layer(model.pool, x, printer, layer_timer, time_list)

    printer.hline()
    display_layer(model, x, printer, total_timer, time_list)
    printer.hline()
    return x, time_list

def display_layer(
    layer:      (torch.nn.Module | str),
    x:          torch.Tensor,
    printer:    util.Printer,
    timer:      util.Timer,
    time_list:  list[float],
):
    time_list.append(timer.get_time_taken())
    t_str = units.time_concise.format(time_list[-1])
    printer( "%-40r -> %-20s in %11s" % (layer, list(x.shape), t_str))

def num_params(layer: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in layer.parameters())

def plot_sequential(
    model:  Sequential,
    x:      torch.Tensor,
) -> plotting.MultiPlot:
    _, t_list   = display_sequential(model, x)
    t_tot       = t_list[-1]
    t_max       = max(t_list[:-1])
    t_tot_label = "Total = %s"  % units.time_concise.format(t_tot)
    t_max_label = "Max = %s"    % units.time_concise.format(t_max)

    layer_list  = [model.embed, *model.layers, model.pool]
    np_list     = [num_params(m) for m in layer_list]
    n_tot_label = "Total = %s"  % units.metric.format(sum(np_list))
    n_max_label = "Max = %s"    % units.metric.format(max(np_list))

    name_list   = [type(m).__name__ for m in layer_list]
    x_plot      = list(range(len(name_list)))
    kwargs      = {
        "xticks":               x_plot,
        "xticklabels":          name_list,
        "rotate_xticklabels":   True,
        "xlabel":               "Layer",
    }
    return plotting.MultiPlot(
        plotting.Subplot(
            plotting.Bar(x_plot, t_list[:-1]),
            plotting.HLine(t_tot, c="k", ls="--", label=t_tot_label),
            plotting.HLine(t_max, c="r", ls="--", label=t_max_label),
            plotting.Legend(),
            **kwargs,
            ylim=[0, 1.1 * t_max],
            ylabel="Time (s)",
        ),
        plotting.Subplot(
            plotting.Bar(x_plot, np_list),
            plotting.HLine(sum(np_list), c="k", ls="--", label=n_tot_label),
            plotting.HLine(max(np_list), c="r", ls="--", label=n_max_label),
            plotting.Legend(),
            **kwargs,
            ylim=[0, 1.1 * max(np_list)],
            ylabel="# Parameters",
        ),
        title="%r\nInput shape = %s" % (model, list(x.shape)),
        figsize=[10, 6],
    )
