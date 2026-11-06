from juml.models.model import Model
from juml.models import layers
from juml.models.layers.linear import LinearLayer
from juml.models.ff import FeedForwardModel
from juml.models.relu_mlp import ReluMlp
from juml.models.rand_relu_features import RandReluFeatures

def get_all_models() -> list[type[Model]]:
    return [
        ReluMlp,
    ]
