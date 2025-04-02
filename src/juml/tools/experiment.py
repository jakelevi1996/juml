from jutility import util, cli
from juml.train.base import Trainer

class Experiment:
    def __init__(self, arg_dict: dict):
        self.arg_dict   = arg_dict
        self.arg_str    = util.format_dict(arg_dict)
        self.model_name = None
        self.ind        = None
        self.metrics    = None
        self.result     = None

    def set_model_name(self, model_name: str):
        self.model_name = model_name

    def set_ind(self, ind: int):
        self.ind = ind

    def load_result(self, args: cli.ParsedArgs, targets: list[str]):
        if self.result is not None:
            raise ValueError(
                "%r already has result=%s"
                % (self, self.result)
            )

        args.update(self.arg_dict)
        self.metrics = util.load_json(Trainer.get_metrics_path(args))

        metric = self.metrics
        for key in targets:
            metric = metric[key]

        result = metric
        if not isinstance(result, float):
            raise ValueError(
                "Target %s in metrics %s has type %s, expected `float`"
                % (targets, self.metrics, type(result))
            )

        self.result = result

    def __repr__(self) -> str:
        return util.format_type(type(self), **self.arg_dict)

    def __lt__(self, other: "Experiment") -> bool:
        return (self.result < other.result)
