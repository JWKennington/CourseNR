import pathlib
import sys

NB_ROOT = pathlib.Path(__file__).parent
PYNSTEIN_ROOT = pathlib.Path(__file__).parent.parent.parent.parent / 'utilities' / 'pystein'


def setup_nb():
	sys.path.append(PYNSTEIN_ROOT.as_posix())
