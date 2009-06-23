"""Very simple test that compares the speed of operating with
aligned vs unaligned arrays.
"""

from timeit import Timer
import numpy as np
import numexpr as ne

niter = 10
#shape = (1000*10000)   # unidimensional test
shape = (1000, 10000)   # multidimensional test

ne.print_versions()

Z_fast = np.zeros(shape, dtype=[('x',np.float64),('y',np.int64)])
Z_slow = np.zeros(shape, dtype=[('x',np.float64),('y',np.bool)])

x_fast = Z_fast['x']
t = Timer("x_fast * x_fast", "from __main__ import x_fast")
print "NumPy aligned-->", round(min(t.repeat(3, niter)), 6)

x_slow = Z_slow['x']
t = Timer("x_slow * x_slow", "from __main__ import x_slow")
print "NumPy unaligned-->", round(min(t.repeat(3, niter)), 6)

t = Timer("ne.evaluate('x_fast * x_fast')", "from __main__ import ne, x_fast")
print "Numexpr aligned-->", round(min(t.repeat(3, niter)), 6)

t = Timer("ne.evaluate('x_slow * x_slow')", "from __main__ import ne, x_slow")
print "Numexpr unaligned-->", round(min(t.repeat(3, niter)), 6)
