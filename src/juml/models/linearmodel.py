
from jutility import cli
from juml.models import embed, pool
from juml.models.sequential import Sequential
from juml.models.linear import Linear

class LinearModel(Sequential):
    def __init__(
        self,
        input_shape:    list[int],
        output_shape:   list[int],
        embedder:       embed.Embedder,
        pooler:         pool.Pooler,
    ):
        self._init_sequential(embedder, pooler)
        self.embed.set_input_shape(input_shape)
        self.pool.set_shapes([], output_shape)

        layer = Linear(
            input_dim=self.embed.get_output_dim(-1),
            output_dim=self.pool.get_input_dim(-1),
        )
        self.layers.append(layer)

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return []
