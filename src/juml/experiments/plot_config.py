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
        sharex:         bool,
        sharey:         bool,
        x_label:        str | None,
        y_label:        str | None,
        c_label:        str | None,
        col_label:      str | None,
        row_label:      str | None,
        ylim:           tuple[float, float] | None,
        figsize:        tuple[float, float],
        font_size:      int,
    ):
        if x_label is None:
            x_label = x_key
        if y_label is None:
            y_label = target_metric
        if c_label is None:
            c_label = c_key
        if col_label is None:
            col_label = col_key
        if row_label is None:
            row_label = row_key

        self.target_metric  = target_metric
        self.x_key          = x_key
        self.c_key          = c_key
        self.col_key        = col_key
        self.row_key        = row_key
        self.log_x          = log_x
        self.log_y          = log_y
        self.sharex         = sharex
        self.sharey         = sharey
        self.x_label        = x_label
        self.y_label        = y_label
        self.c_label        = c_label
        self.col_label      = col_label
        self.row_label      = row_label
        self.ylim           = ylim
        self.figsize        = figsize
        self.font_size      = font_size
        self.cp             = plotting.ColourPicker.contrast()

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
            cli.Arg("sharex",           action="store_true"),
            cli.Arg("sharey",           action="store_true"),
            cli.Arg("x_label",          type=str, default=None),
            cli.Arg("y_label",          type=str, default=None),
            cli.Arg("c_label",          type=str, default=None),
            cli.Arg("col_label",        type=str, default=None),
            cli.Arg("row_label",        type=str, default=None),
            cli.Arg("ylim",             type=float, default=None,   nargs=2),
            cli.Arg("figsize",          type=float, default=[6, 4], nargs=2),
            cli.Arg("font_size",        type=int,   default=12),
            is_group=True,
        )
