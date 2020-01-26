# References:
#
# http://software.intel.com/en-us/intel-mkl
# https://github.com/pydata/numexpr/wiki/NumexprMKL

from __future__ import print_function
import datetime
import sys
import numpy as np
import numexpr as ne
from time import time

N = int(2**26)

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
z = np.empty(N, dtype=np.float64)

# Our working set is 3 vectors of N doubles each
working_set_GB = 3 * N * 8 / 2**30

print("NumPy version: %s" % (np.__version__,))

t0 = time()
z = 2*y + 4*x
t1 = time()
gbs = working_set_GB / (t1-t0)
print("Time for an algebraic expression:     %.3f s / %.3f GB/s" % (t1-t0, gbs))

t0 = time()
z = np.sin(x)**2 + np.cos(y)**2
t1 = time()
gbs = working_set_GB / (t1-t0)
print("Time for a transcendental expression: %.3f s / %.3f GB/s" % (t1-t0, gbs))

if ne.use_vml:
    ne.set_vml_num_threads(1)
    ne.set_num_threads(8)
    print("NumExpr version: %s, Using MKL ver. %s, Num threads: %s" % (ne.__version__, ne.get_vml_version(), ne.nthreads))
else:
    ne.set_num_threads(8)
    print("NumExpr version: %s, Not Using MKL, Num threads: %s" % (ne.__version__, ne.nthreads))

t0 = time()
ne.evaluate('2*y + 4*x', out = z)
t1 = time()
gbs = working_set_GB / (t1-t0)
print("Time for an algebraic expression:     %.3f s / %.3f GB/s" % (t1-t0, gbs))

t0 = time()
ne.evaluate('sin(x)**2 + cos(y)**2', out = z)
t1 = time()
gbs = working_set_GB / (t1-t0)
print("Time for a transcendental expression: %.3f s / %.3f GB/s" % (t1-t0, gbs))
