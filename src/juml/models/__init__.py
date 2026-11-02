from juml.models.model import Model
from juml.models.ff import FeedForwardModel
from juml.models.relu_mlp import ReluMlp
from juml.models import layers
from juml.models.layers.linear import LinearLayer

def get_all_models() -> list[type[Model]]:
    return [
        ReluMlp,
    ]
