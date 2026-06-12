from abc import ABC, abstractmethod


class Command(ABC):
    name: str
    help: str

    @classmethod
    def add_parser(cls, subparsers):
        parser = subparsers.add_parser(cls.name, help=cls.help)
        cls.add_args(parser)
        parser.set_defaults(script=cls)

    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def __call__(self):
        ...
