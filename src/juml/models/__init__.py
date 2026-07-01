from juml.models import layers
from juml.models.model import Model
from juml.models.classification import FeedForwardModel
from juml.models.relu_mlp import ReluMlp

def get_all_models() -> list[type[Model]]:
    return [
        ReluMlp,
    ]
