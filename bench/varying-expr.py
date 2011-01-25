# Benchmark for checking if numexpr leaks memory when evaluating
# expressions that changes continously.  It also serves for computing
# the latency of numexpr when working with small arrays.

import sys
from time import time
import numpy as np
import numexpr as ne

N = 1000*10
M = 1000

print "Number of iterations %s.  Length of the array: %s " % (N, M)

a = np.arange(M)

print "Expressions that *cannot* be cached:"
t1 = time()
for i in xrange(N):
    r = ne.evaluate("2 * a + (a + 1) ** 2 == %d" % i)
    if i % 1000 == 0:
        sys.stdout.write('.')
print "\nEvaluated %s iterations in: %s seconds" % (N, round(time()-t1,3))

print "Expressions that *can* be cached:"
t1 = time()
for i in xrange(N):
    r = ne.evaluate("2 * a + (a + 1) ** 2 == i")
    if i % 1000 == 0:
        sys.stdout.write('.')
print "\nEvaluated %s iterations in: %s seconds" % (N, round(time()-t1,3))

print "Using python virtual machine with non-cacheable expressions:"
t1 = time()
for i in xrange(N):
    r = eval("2 * a + (a + 1) ** 2 == %d" % i)
    if i % 1000 == 0:
        sys.stdout.write('.')
print "\nEvaluated %s iterations in: %s seconds" % (N, round(time()-t1,3))
