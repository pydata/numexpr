###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: BSD
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

"""
Numexpr3 is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (such as "3*a+4*b") are multi-threaded 
and blocked within a virtual machine to achieve memory-bound computing 
performance and less memory than doing the same calculation in NumPy.

See:

https://github.com/pydata/numexpr

and 

https://numexpr.readthedocs.io

for more information.
"""

import unittest

from .ne3compiler import NumExpr, evaluate, OPTABLE, wisdom
from .interpreter import MAX_ARGS, set_tempsize
from .utils import get_ncores, get_nthreads, set_nthreads, print_info, str_info

def print_versions():
    """Print the versions of software that numexpr relies on."""
    try:
        import numexpr.tests
        return numexpr.tests.print_versions()
    except ImportError:
        # To maintain Python 2.6 compatibility we have simple error handling
        raise ImportError('`numexpr.tests` could not be imported, likely it was excluded from the distribution.')

def test(verbosity: int=1) -> unittest.TextTestResult:
    '''
    Run the test suite.

    This function is a stub that redirects to `numexpr3.tests.test_numexpr.test()`
    '''
    try:
        from .tests import test
    except ImportError as e:
        raise ImportError('Could not import `numexpr3.tests`, was it included with the distribution?') from e
    else:
        return test(verbosity=verbosity)


from . import __version__
__version__ = __version__.__version__

