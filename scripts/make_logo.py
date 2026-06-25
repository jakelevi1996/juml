import numpy as torch
from jutility import plotting

num_colours = 7
x = torch.linspace(-1, 7, 100)
cp = plotting.ColourPicker.from_linear_cmap("plasma", num_colours)
lines = [
    plotting.Line(x, ((1 + (i/10)) * torch.sin(x + (i / num_colours))), c=c)
    for i, c in enumerate(cp)
]
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
