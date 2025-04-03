import statistics
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

class ExperimentGroup:
    def __init__(
        self,
        params:         dict[str, list],
        seeds:          list[int],
        experiments:    list[Experiment],
    ):
        self.params             = params
        self.seeds              = seeds
        self.experiment_list    = experiments
        self.experiment_dict    = {e.arg_str: e for e in experiments}
        self.results            = [e.result     for e in experiments]

    @classmethod
    def from_params(
        cls,
        params: dict[str, list],
        seeds:  list[int],
    ):
        components_list = [[["seed", s]] for s in seeds]
        for param_name, param_vals in params.items():
            components_list = [
                c + p
                for c in components_list
                for p in [[[param_name, v]] for v in param_vals]
            ]

        return cls(
            params=params,
            seeds=seeds,
            experiments=[
                Experiment({k: v for k, v in c})
                for c in components_list
            ],
        )

    def load_results(self, args: cli.ParsedArgs, targets: list[str]):
        for e in self.experiment_list:
            e.load_result(args, targets)

        self.results = [e.result for e in self.experiment_list]

    def sweep_seeds(
        self,
        root_experiment: Experiment,
    ) -> "ExperimentGroup":
        experiment_list = []
        root_dict       = root_experiment.arg_dict
        root_seed       = root_dict["seed"]

        for seed in self.seeds:
            root_dict["seed"] = seed
            arg_str = util.format_dict(root_dict)
            experiment_list.append(self.experiment_dict[arg_str])

        root_dict["seed"] = root_seed
        return ExperimentGroup(
            params=dict(),
            seeds=self.seeds,
            experiments=experiment_list,
        )

    def sweep_param(
        self,
        root_experiment:    Experiment,
        param_name:         str,
    ) -> "ExperimentGroup":
        experiment_list = []
        root_dict       = root_experiment.arg_dict
        root_val        = root_dict[param_name]
        root_seed       = root_dict["seed"]

        for val in self.params[param_name]:
            for seed in self.seeds:
                root_dict[param_name]   = val
                root_dict["seed"]       = seed
                arg_str = util.format_dict(root_dict)
                experiment_list.append(self.experiment_dict[arg_str])

        root_dict[param_name]   = root_val
        root_dict["seed"]       = root_seed
        return ExperimentGroup(
            params=dict(),
            seeds=self.seeds,
            experiments=experiment_list,
        )

    def results_mean(self) -> float:
        return statistics.mean(self.results)

    def results_std(self) -> float:
        return statistics.stdev(self.results)

    def __iter__(self):
        return iter(self.experiment_list)

    def __len__(self):
        return len(self.experiment_list)
