from jutility import cli, util
from juml.commands import get_all_commands
from juml.commands.command import Command

class Framework:
    @classmethod
    def get_commands(cls) -> list[type[Command]]:
        return get_all_commands()

    @classmethod
    def get_parser(cls):
        return cli.Parser(
            sub_commands=cli.SubCommandGroup(
                *[
                    command_type.init_framework_command()
                    for command_type in cls.get_commands()
                ],
            ),
        )

    @classmethod
    def get_command(cls, *parser_args, **parser_kwargs) -> Command:
        parser = cls.get_parser()
        args = parser.parse_args(*parser_args, **parser_kwargs)
        command = args.get_command()
        return command

    @classmethod
    def run(cls, *parser_args, **parser_kwargs):
        command = cls.get_command(*parser_args, **parser_kwargs)
        with util.Timer(repr(command)):
            command.run(**command.get_kwargs())
