# Small benchmark to get the even point where the threading code
# performs better than the serial code.  See issue #36 for details.

from __future__ import print_function
import numpy as np
import numexpr as ne
from numpy.testing import assert_array_equal
from time import time

def bench(N):
    print("*** array length:", N)
    a = np.arange(N)
    t0 = time()
    ntimes = (1000*2**15) // N
    for i in range(ntimes):
        ne.evaluate('a>1000')
    print("numexpr--> %.3g" % ((time()-t0)/ntimes,))

    t0 = time()
    for i in range(ntimes):
        eval('a>1000')
    print("numpy--> %.3g" % ((time()-t0)/ntimes,))

if __name__ == "__main__":
    print("****** Testing with 1 thread...")
    ne.set_num_threads(1)
    for N in range(10, 20):
        bench(2**N)

    print("****** Testing with 2 threads...")
    ne.set_num_threads(2)
    for N in range(10, 20):
        bench(2**N)

