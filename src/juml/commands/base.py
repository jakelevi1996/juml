from jutility import cli

class Command(cli.SubCommand):
    def run(self, args: cli.ParsedArgs):
        raise NotImplementedError()

    @classmethod
    def init_juml(cls, train_args: list[cli.Arg]):
        return cls(
            cls.get_name(),
            *cls.get_args(train_args),
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def get_args(cls, train_args: list[cli.Arg]) -> list[cli.Arg]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.get_name()
