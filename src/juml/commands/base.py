from jutility import cli

class Command(cli.SubCommand):
    def run(
        self,
        args: cli.ParsedArgs,
        **kwargs,
    ):
        raise NotImplementedError()

    @classmethod
    def init_juml(cls, train_args: list[cli.Arg]):
        return cls(
            cls.get_name(),
            *cls.select_train_args(train_args),
            cli.ArgGroup(
                cls.get_name(),
                *cls.get_cli_options(),
            ),
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def select_train_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        return train_args

    @classmethod
    def get_cli_options(cls) -> list[cli.Arg]:
        return []

    def __repr__(self) -> str:
        return self.get_name()
