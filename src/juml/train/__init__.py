from juml.train.base import Trainer
from juml.train.bpsp import BpSp
from juml.train.sweeper import Sweeper
from juml.train.profiler import Profiler

def get_all() -> list[type[Trainer]]:
    return [
        BpSp,
    ]
