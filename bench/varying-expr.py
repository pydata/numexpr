# Benchmark for checking if numexpr leaks memory when evaluating
# expressions that changes continously.  It also serves for computing
# the latency of numexpr when working with small arrays.

import sys
from time import time
import numpy as np
import numexpr as ne

N = 1000*100
M = 1000

print "Number of iterations %s.  Length of the array: %s " % (N, M)

a = np.arange(M)

t1 = time()
for i in xrange(N):
    #r = ne.evaluate("a == i")        # expression is cached
    r = ne.evaluate("a == %d" % i)   # expression cannot be cached
    if i % 1000 == 0:
        sys.stdout.write('.')

print "\nEvaluated %s iterations in: %s seconds" % (N, round(time()-t1,3))
