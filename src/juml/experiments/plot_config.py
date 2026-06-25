from jutility import plotting, cli

class PlottingConfig:
    def __init__(
        self,
        target_metric:  str | None,
        x_key:          str | None,
        c_key:          str | None,
        col_key:        str | None,
        row_key:        str | None,
        log_x:          bool,
        log_y:          bool,
        sharey:         bool,
        ylim:           tuple[float, float] | None,
        figsize:        tuple[float, float],
        font_size:      int,
        abbreviations:  dict[str, str],
    ):
        self.target_metric  = target_metric
        self.x_key          = x_key
        self.c_key          = c_key
        self.col_key        = col_key
        self.row_key        = row_key
        self.log_x          = log_x
        self.log_y          = log_y
        self.sharey         = sharey
        self.ylim           = ylim
        self.figsize        = figsize
        self.font_size      = font_size
        self.cp             = plotting.ColourPicker.contrast()
        self.abbreviations  = abbreviations

    def abbrev(self, s: str | None) -> str | None:
        return self.abbreviations.get(s, s)

    @classmethod
    def get_cli_arg(cls) -> cli.ObjectArg:
        return cli.ObjectArg(
            cls,
            cli.Arg("target_metric",    type=str, default=None),
            cli.Arg("x_key",            type=str, default=None),
            cli.Arg("c_key",            type=str, default=None),
            cli.Arg("col_key",          type=str, default=None),
            cli.Arg("row_key",          type=str, default=None),
            cli.Arg("log_x",            action="store_true"),
            cli.Arg("log_y",            action="store_true"),
            cli.Arg("sharey",           action="store_true"),
            cli.Arg("ylim",             type=float, default=None,   nargs=2),
            cli.Arg("figsize",          type=float, default=[6, 4], nargs=2),
            cli.Arg("font_size",        type=int,   default=12),
            cli.JsonArg(
                "abbreviations",
                default=dict(),
                help=(
                    "Optional abbreviations for axis labels and legend and "
                    "subplot titles, as a dictionary mapping strings to "
                    "their abbreviations, in a JSON string of the form "
                    "'{\"a1\": \"s1\", \"a2,a3\": \"s2\", ...}'."
                ),
            ),
            is_group=True,
        )
