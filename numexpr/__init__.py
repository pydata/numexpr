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

from __config__ import show as show_config, get_info

if get_info('mkl'):
    use_vml = True
else:
    use_vml = False

from cpuinfo import cpu
if cpu.is_AMD() or cpu.is_Intel():
    is_cpu_amd_intel = True
else:
    is_cpu_amd_intel = False

import os, os.path
import platform
from numexpr.expressions import E
from numexpr.necompiler import NumExpr, disassemble, evaluate
from numexpr.tests import test, print_versions
from numexpr.utils import (
    get_vml_version, set_vml_accuracy_mode, set_vml_num_threads,
    set_num_threads, detect_number_of_cores)

# Detect the number of cores
ncores = detect_number_of_cores()

# Initialize the number of threads to be used
# If this is modified, please update the note in:
# https://github.com/pydata/numexpr/wiki/Numexpr-Users-Guide
try:
    nthreads = int(os.environ['NUMEXPR_NUM_THREADS'])
except KeyError:
    nthreads = ncores
    # Check that we don't activate too many threads at the same time.
    # 8 seems a sensible value.
    if nthreads > 8:
        nthreads = 8
# Check that we don't surpass the MAX_THREADS in interpreter.cpp
if nthreads > 4096:
    nthreads = 4096
if 'sparc' in platform.machine():
    import warnings
    warnings.warn('The number of threads have been set to 1 because problems related '
                  'to threading have been reported on some sparc machine. '
                  'The number of threads can be changes using the "set_num_threads" '
                  'function.')
    set_num_threads(1)
else:
    set_num_threads(nthreads)

# The default for VML is 1 thread (see #39)
set_vml_num_threads(1)

import version

dirname = os.path.dirname(__file__)

__version__ = version.version

