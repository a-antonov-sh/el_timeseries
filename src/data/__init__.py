from data.data import Data
from common.command import Command


def _run_prepare():
    data = Data()
    data.load_data()
    data.save_prepared()


class PrepareCommand(Command):
    name = "prepare"
    help = "Load dataset and save prepared parquet files"

    def __call__(self):
        _run_prepare()