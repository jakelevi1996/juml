import numpy as torch
from jutility import plotting

num_colours = 7
x = torch.linspace(-1, 7, 100)
lines = [
    plotting.Line(x, ((1 + (i/10)) * torch.sin(x + (i / num_colours))))
    for i in range(num_colours)
]
cp = plotting.ColourPicker(num_colours, cyclic=False, cmap_name="plasma")
cp.colourise(lines)
mp = plotting.MultiPlot(
    plotting.Subplot(
        *lines,
        axis_off=True,
        xlim=[-1, 7],
    ),
    figsize=[10, 4],
    colour="k",
    title="  ".join("JUML"),
    title_colour="w",
    title_font_size=40,
)
mp.save("logo_black", dir_name="scripts/img")
