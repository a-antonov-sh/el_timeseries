from argparse import ArgumentParser


def program(name, description, commands):
    parser = ArgumentParser(prog=name, description=description)
    subparsers = parser.add_subparsers(help='Command to execute', required=True)
    for cmd in commands:
        cmd.add_parser(subparsers)
    args = parser.parse_args()
    args.script(args)()