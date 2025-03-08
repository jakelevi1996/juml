from jutility import cli

class Command(cli.SubCommand):
    def run(self, args: cli.ParsedArgs):
        raise NotImplementedError()

    @classmethod
    def init_juml(cls, train_arg: cli.ObjectArg):
        return cls(
            cls.get_name(),
            *cls.get_args(train_arg),
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def get_args(cls, train_arg: cli.ObjectArg) -> list[cli.Arg]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.get_name()
