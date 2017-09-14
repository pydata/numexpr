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

for more infomotion.
"""

from __future__ import absolute_import

from .__config__ import show as _show_config
from .__config__ import get_info as _get_info

from numexpr3.ne3compiler import NumExpr, evaluate, OPTABLE, wisdom
from numexpr3.interpreter import MAX_ARGS, set_tempsize
from numexpr3.tests import test

# Do not import cpuinfo, it needs a major overhaul to fix various 
# multiprocessing bugs on Windows
# from numexpr3.cpuinfo import get_cpu_info
from numexpr3.utils import get_ncores, get_nthreads, set_nthreads, print_info, str_info


from . import __version__
__version__ = __version__.__version__

