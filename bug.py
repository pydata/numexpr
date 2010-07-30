import numpy as np
import numexpr as ne
from time import time

N = 1e7

#a = np.arange(N, dtype='f8')
#a = np.empty(N, dtype='f8')
#a = np.linspace(-1, 1, N).reshape(1000, 10000)
a = np.linspace(-1, 1, N)
#a = np.linspace(-1, 1, N).astype('f4')

# t0 = time()
# #b = np.sin(a)
# b = ((.25*a + .75)*a - 1.5)*a - 2
# #b = a**3
# #b = a.copy()
# print "Time numpy: %.3f" % (time()-t0)

for nthreads in range(6):
    ne.set_num_threads(nthreads+1)
    t0 = time()
    #c = ne.evaluate("sin(a)")
    c = ne.evaluate("((.25*a + .75)*a - 1.5)*a - 2")
    # c = ne.evaluate("sin(a)**2+cos(a)**2")
    # c = ne.evaluate("a**3")
    # c = ne.evaluate("a")
    print "Time numexpr with %d threads: %.3f" % (nthreads+1, time()-t0)

# print "b-->", b
# print "c-->", c
assert np.allclose(b,c)
