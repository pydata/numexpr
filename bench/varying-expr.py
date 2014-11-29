###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

# Benchmark for checking if numexpr leaks memory when evaluating
# expressions that changes continously.  It also serves for computing
# the latency of numexpr when working with small arrays.

from __future__ import print_function
import sys
from time import time
import numpy as np
import numexpr as ne

N = 100
M = 10

def timed_eval(eval_func, expr_func):
    t1 = time()
    for i in range(N):
        r = eval_func(expr_func(i))
        if i % 10 == 0:
            sys.stdout.write('.')
    print(" done in %s seconds" % round(time() - t1, 3))

print("Number of iterations %s.  Length of the array: %s " % (N, M))

a = np.arange(M)

# lots of duplicates to collapse
#expr = '+'.join('(a + 1) * %d' % i for i in range(50))
# no duplicate to collapse
expr = '+'.join('(a + %d) * %d' % (i, i) for i in range(50))

def non_cacheable(i):
    return expr + '+ %d' % i

def cacheable(i):
    return expr + '+ i'

print("* Numexpr with non-cacheable expressions: ", end=" ")
timed_eval(ne.evaluate, non_cacheable)
print("* Numexpr with cacheable expressions: ", end=" ")
timed_eval(ne.evaluate, cacheable)
print("* Numpy with non-cacheable expressions: ", end=" ")
timed_eval(eval, non_cacheable)
print("* Numpy with cacheable expressions: ", end=" ")
timed_eval(eval, cacheable)
