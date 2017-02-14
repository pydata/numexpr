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
Numexpr is a fast numerical expression evaluator for NumPy.  With it,
expressions that operate on arrays (like "3*a+4*b") are accelerated
and use less memory than doing the same calculation in Python.

See:

https://github.com/pydata/numexpr

for more info about it.
"""

from __config__ import show as _show_config
from __config__ import get_info as _get_info


<<<<<<< HEAD:numexpr/__init__.py
if cpu.is_AMD() or cpu.is_Intel():
    is_cpu_amd_intel = True
else:
    is_cpu_amd_intel = False

import os, os.path
import platform
from numexpr.expressions import E
from numexpr.necompiler import NumExpr, disassemble, evaluate, re_evaluate
from numexpr.tests import test, print_versions
from numexpr.utils import (
    get_vml_version, set_vml_accuracy_mode, set_vml_num_threads,
=======
from numexpr3.ne3compiler import NumExpr, evaluate, OPTABLE, wisdom
from numexpr3.utils import (
    print_info, 
>>>>>>> 3b5260be1d8bdf82b50269b47799b508c6715348:numexpr3/__init__.py
    set_num_threads, detect_number_of_cores, detect_number_of_threads)
try: # VML functions are no longer created if NumExpr was not compiled with VML
    from numexpr.utils import ( 
            get_vml_version, set_vml_accuracy_mode, set_vml_num_threads )
except ImportError: pass

# Detect the number of cores
# RAM: the functions in util doesn't update numexpr.ncores or numexpr.nthreads, 
ncores = detect_number_of_cores()
nthreads = detect_number_of_threads()

# Initialize the number of threads to be used
import platform as __platform
if 'sparc' in __platform.machine():
    import warnings as __warnings

    __warnings.warn('The number of threads have been set to 1 because problems related '
                  'to threading have been reported on some sparc machine. '
                  'The number of threads can be changed using the "set_num_threads" '
                  'function.')
    set_num_threads(1)
else:
    set_num_threads(nthreads)

# The default for VML is 1 thread (see #39)
set_vml_num_threads(1)

from . import __version__
__version__ = __version__.__version__

