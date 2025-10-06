###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

"""
Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.

See:

https://github.com/pydata/numexpr

for more info about it.

"""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import unittest

# the `import _ as _` are needed for mypy to understand these are re-exports

from numexpr.interpreter import __BLOCK_SIZE1__ as __BLOCK_SIZE1__
from numexpr.interpreter import MAX_THREADS as MAX_THREADS
from numexpr.interpreter import use_vml as use_vml

is_cpu_amd_intel: Final = False # DEPRECATION WARNING: WILL BE REMOVED IN FUTURE RELEASE

# cpuinfo imports were moved into the test submodule function that calls them
# to improve import times.

from numexpr.expressions import E as E
from numexpr.necompiler import NumExpr as NumExpr
from numexpr.necompiler import disassemble as disassemble
from numexpr.necompiler import evaluate as evaluate
from numexpr.necompiler import re_evaluate as re_evaluate
from numexpr.necompiler import validate as validate
from numexpr.utils import _init_num_threads
from numexpr.utils import detect_number_of_cores as detect_number_of_cores
from numexpr.utils import detect_number_of_threads as detect_number_of_threads
from numexpr.utils import get_num_threads as get_num_threads
from numexpr.utils import get_vml_version as get_vml_version
from numexpr.utils import set_num_threads as set_num_threads
from numexpr.utils import set_vml_accuracy_mode as set_vml_accuracy_mode
from numexpr.utils import set_vml_num_threads as set_vml_num_threads

# Detect the number of cores
ncores: Final = detect_number_of_cores()
# Initialize the number of threads to be used
nthreads: Final = _init_num_threads()
# The default for VML is 1 thread (see #39)
# set_vml_num_threads(1)

from . import version as version

__version__: Final = version.version

def print_versions() -> None:
    """Print the versions of software that numexpr relies on."""
    try:
        import numexpr.tests
        return numexpr.tests.print_versions()  # type: ignore[no-untyped-call]
    except ImportError:
        # To maintain Python 2.6 compatibility we have simple error handling
        raise ImportError('`numexpr.tests` could not be imported, likely it was excluded from the distribution.')

def test(verbosity: int = 1) -> "unittest.result.TestResult":
    """Run all the tests in the test suite."""
    try:
        import numexpr.tests
        return numexpr.tests.test(verbosity=verbosity)  # type: ignore[no-untyped-call]
    except ImportError:
        # To maintain Python 2.6 compatibility we have simple error handling
        raise ImportError('`numexpr.tests` could not be imported, likely it was excluded from the distribution.')
